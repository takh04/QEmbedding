import data
import torch
import Hybrid_nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Uisng Device: {device}\n")

#batch_size = 25
#iterations = 1000
batch_size = 5
iterations = 40

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
#X1_train_distance, X0_train_distance = torch.tensor(X1_train).to(device), torch.tensor(X0_train).to(device)
#X1_valid_distance, X0_valid_distance = torch.tensor(X1_test[:300]).to(device), torch.tensor(X0_test[:300]).to(device)
#X1_test_distance, X0_test_distance = torch.tensor(X1_test[300:]).to(device), torch.tensor(X0_test[300:]).to(device)
X1_train_distance, X0_train_distance = torch.tensor(X1_train).to(device), torch.tensor(X0_train).to(device)
X1_valid_distance, X0_valid_distance = torch.tensor(X1_test[:10]).to(device), torch.tensor(X0_test[:10]).to(device)
X1_test_distance, X0_test_distance = torch.tensor(X1_test[10:20]).to(device), torch.tensor(X0_test[10:20]).to(device)

#make new data for hybrid model
def new_data(batch_size, X, Y):
    X1_new, X2_new, Y_new = [], [], []
    for i in range(batch_size):
        n, m = np.random.randint(len(X)), np.random.randint(len(X))
        X1_new.append(X[n])
        X2_new.append(X[m])
        if Y[n] == Y[m]:
            Y_new.append(1)
        else:
            Y_new.append(0)
    return torch.tensor(X1_new).to(device), torch.tensor(X2_new).to(device), torch.tensor(Y_new).to(device)

#N_valid, N_test = 500, 10000
N_valid, N_test = 10, 10
X1_new_valid, X2_new_valid, Y_new_valid = new_data(N_valid, X_test, Y_test)
X1_new_test, X2_new_test, Y_new_test = new_data(N_test, X_test, Y_test)


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
    correct90, correct80 = 0, 0
    for p,l in zip(predictions, labels):
        if torch.abs(p - l) < 0.1:
            correct90 += 1
        elif torch.abs(p - l) < 0.2:
            correct80 += 1
    return correct90 / len(predictions) * 100, (correct80 + correct90) / len(predictions) * 100, 


# Train model1 and model2
def train_models(model_name):
    train_loss, valid_loss, valid_acc90, valid_acc80 = [], [], [], []
    model = Hybrid_nn.get_model(model_name).to(device)
    model.train()

    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    for it in range(iterations):
        X1_batch, X2_batch, Y_batch = new_data(batch_size, X_train, Y_train)
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
                accuracy90_validation, accuracy80_validation = accuracy(pred_validation, Y_new_valid)
                valid_loss.append(loss_validation.item())
                valid_acc90.append(accuracy90_validation)
                valid_acc80.append(accuracy80_validation)
                print(f"Validation Loss: {loss_validation}, Validation Accuracy (>0.9): {accuracy90_validation}%, Validation Accuracy (>0.8): {accuracy80_validation}%")
                if early_stopper.early_stop(loss_validation):
                    print("Loss converged!")
                    break
    
    with torch.no_grad():
        pred_test = model(X1_new_test, X2_new_test)
        loss_test = loss_fn(pred_test, Y_new_test)
        accuracy90_test, accuracy80_test = accuracy(pred_test, Y_new_test)
        print(f"Test Loss: {loss_test}, Test Accuracy (>0.9): {accuracy90_test}%, Test Accuracy (>0.8): {accuracy80_test}%")

    f = open(f"Results/{model_name}.txt", 'w')
    f.write("Loss History:\n")
    f.write(str(train_loss))
    f.write("\n\n")
    f.write("Validation Loss History:\n")
    f.write(str(valid_loss))
    f.write("\n")
    f.write("Validation Accuracy90 History:\n")
    f.write(str(valid_acc90))
    f.write("\n\n")
    f.write("Validation Accuracy80 History:\n")
    f.write(str(valid_acc80))
    f.write("\n\n")
    f.write(f"Test Loss: {loss_test}\n")
    f.write(f"Test Accuracy90: {accuracy90_test}\n")
    f.write(f"Test Accuracy80: {accuracy80_test}\n")
    f.close()
    torch.save(model.state_dict(), f'Results/{model_name}.pt')


# Train distance model1 and distance model2
def train_distance_models(model_name):
    train_loss, valid_loss = [], []
    model = Hybrid_nn.get_model(model_name).to(device)
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

    f = open(f"Results/{model_name}.txt", 'w')
    f.write("Loss History:\n")
    f.write(str(train_loss))
    f.write("\n\n")
    f.write("Validation Loss History:\n")
    f.write(str(valid_loss))
    f.write("\n\n")
    f.write(f"Test Loss: {loss_test.item()}\n")
    f.close()
    torch.save(model.state_dict(), f'Results/{model_name}.pt')


def train(model_names):
    for model_name in model_names:
        if model_name in ['Model1_Fidelity', 'Model1_HSinner', 'Model2_Fidelity', 'Model2_HSinner']:
            train_models(model_name)
        elif model_name in ['DistanceModel1_Trace', 'DistanceModel1-HS', 'DistanceModel2_Trace', 'DistanceModel2_HS']:
            train_distance_models(model_name)

model_names = ['Model1_Fidelity', 'Model1_HSinner', 'Model2_Fidelity', 'Model2_HSinner', 'DistanceModel1_Trace', 'DistanceModel1_HS', 'DistanceModel2_Trace', 'DistanceModel2_HS']
train(model_names)