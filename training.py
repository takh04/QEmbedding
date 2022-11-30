import data
import torch
import parameters
import Hybrid_nn
import numpy as np

device = parameters.device
print(f"Uisng Device: {device}\n")

batch_size = 25
iterations = 1000

#load data
feature_reduction = 'PCA8'
classes = [0,1]
X_train, X_test, Y_train, Y_test = data.data_load_and_process('mnist', '2', feature_reduction=feature_reduction, classes=classes)

X1_train, X0_train, X1_test, X0_test = [], [], [], []
for i in range(len(X_train)):
    if Y_train[i] == 1:
        X1_train.append(X_train[i])
    else:
        X0_train.append(X_train[i])
for i in range(len(X_test)):
    if Y_test[i] == 1:
        X1_test.append(X_test[i])
    else:
        X0_test.append(X_test[i])

# train, validation, test dataset for hybrid distance model
X1_train_distance, X0_train_distance = torch.tensor(X1_train).to(device), torch.tensor(X0_train).to(device)
X1_valid_distance, X0_valid_distance = torch.tensor(X1_test[:300]).to(device), torch.tensor(X0_test[:300]).to(device)
X1_test_distance, X0_test_distance = torch.tensor(X1_test[300:]).to(device), torch.tensor(X0_test[300:]).to(device)

#make new data for hybrid model
def new_data(batch_size):
    X1_new, X2_new, Y_new = [], [], []
    for i in range(batch_size):
        n, m = np.random.randint(len(X_train)), np.random.randint(len(X_train))
        X1_new.append(X_train[n])
        X2_new.append(X_train[m])
        if Y_train[n] == Y_train[m]:
            Y_new.append(1)
        else:
            Y_new.append(0)
    return torch.tensor(X1_new), torch.tensor(X2_new), torch.tensor(Y_new)

N_valid, N_test = 300, 1000
X1_new_valid, X2_new_valid, Y_new_valid = new_data(N_valid)
X1_new_test, X2_new_test, Y_new_test = new_data(N_test)


# Early Stopping and Accuracy
class EarlyStoppter:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

early_stopper = EarlyStoppter(patience=3, min_delta=0)

def accuracy(predictions, labels):
    correct = 0
    for p,l in zip(predictions, labels):
        if np.abs(p - 1) < 0.1:
            correct += 1
    return correct / len(predictions) * 100


# Train Hybrid models
def train_hybrid():
    train_loss, valid_loss, valid_acc = [], [], []
    model = Hybrid_nn.get_model().to(device)
    model.train()

    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    for it in range(iterations):
        X1_batch, X2_batch, Y_batch = new_data(batch_size)
        X1_batch, X2_batch, Y_batch = X1_batch.to(device), X2_batch.to(device), Y_batch.to(device)

        pred = model(X1_batch, X2_batch)
        pred, Y_batch = pred.to(torch.float32), Y_batch.to(torch.float32)
        loss = loss_fn(pred, Y_batch)
        train_loss.append(loss.item())

        opt.zero_grad()
        loss.backward()
        opt.step()

        if it % 20 == 0:
            print(f"Iterations: {it} Loss: {loss.item()}")
            with torch.no_grad():
                pred_validation = model(X1_new_valid, X2_new_valid)
                loss_validation = loss_fn(pred_validation, Y_new_valid)
                accuracy_validation = accuracy(pred_validation, Y_new_valid)
                valid_loss.append(loss_validation.item())
                valid_acc.append(accuracy_validation)
                print(f"Validation Loss: {loss_validation}, Validation Accuracy: {accuracy_validation}%")
                if early_stopper.early_stop(loss_validation):
                    print("Loss converged!")
                    break
    
    with torch.no_grad():
        pred_test = model(X1_new_test, X2_new_test)
        loss_test = loss_fn(pred_test, Y_new_test)
        accuracy_test = accuracy(pred_test, Y_new_test)
        print(f"Test Loss: {loss_test}, Test Accuracy: {accuracy_test}%")

    f = open(f"Results/{parameters.model} + {parameters.measure}.txt", 'w')
    f.write("Loss History:\n")
    f.write(str(train_loss))
    f.write("\n\n")
    f.write("Validation Loss History:\n")
    f.write(str(valid_loss))
    f.write("\n")
    f.write("Validation Accuracy History:\n")
    f.write(str(valid_acc))
    f.write("\n\n")
    f.write(f"Test Loss: {loss_test}\n")
    f.write(f"Test Accuracy: {accuracy_test}\n")
    f.close()
    torch.save(model.state_dict(), f'Results/{parameters.model} + {parameters.measure}.pt')


# Train distance models
def train_hybrid_distance():
    train_loss, valid_loss = [], []
    model = Hybrid_nn.get_model().to(device)
    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for it in range(iterations):
        
        batch_index = np.random.randint(0, min(len(X1_train_distance), len(X0_train_distance)), (batch_size,))
        X1_batch, X0_batch = X1_train_distance[batch_index], X0_train_distance[batch_index]
        X1_batch, X0_batch = torch.tensor(X1_batch).to(device), torch.tensor(X0_batch).to(device)

        loss = model(X1_batch, X0_batch)
        train_loss.append(loss.item())

        opt.zero_grad()
        loss.backward()
        opt.step()

        if it % 20 == 0:
            print(f"Iterations: {it} Loss: {loss.item()}")
            with torch.no_grad():
                loss_validation = model(X1_valid_distance, X0_valid_distance)
                valid_loss.append(loss_validation.item())
                print(f"Validation Loss: {loss_validation}")
                if early_stopper.early_stop(loss_validation):
                    print("Loss converged!")
                    break

    with torch.no_grad():
        loss_test = model(X1_test_distance, X0_test_distance)
        print(f"Test Loss: {loss_test.item()}")

    f = open(f"Results/{parameters.model} + {parameters.measure}.txt", 'w')
    f.write("Loss History:\n")
    f.write(str(train_loss))
    f.write("\n\n")
    f.write("Validation Loss History:\n")
    f.write(str(valid_loss))
    f.write("\n\n")
    f.write(f"Test Loss: {loss_test.item()}\n")
    f.close()
    torch.save(model.state_dict(), f'Results/{parameters.model} + {parameters.measure}.pt')

def train():
    if parameters.model == 'model1' or parameters.model == 'model2':
        train_hybrid()
    else:
        train_hybrid_distance()

train()