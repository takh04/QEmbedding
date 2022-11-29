import data
import torch
import parameters
import Hybrid_nn
import numpy as np

device = parameters.device
print(f"Uisng Device: {device}\n")

batch_size = 50
iterations = 500

#load data
feature_reduction = 'PCA8'
classes = [0,1]
X_train, X_test, Y_train, Y_test = data.data_load_and_process('mnist', '2', feature_reduction=feature_reduction, classes=classes)

trainfilter0, trainfilter1 = np.where((Y_train == classes[0])), np.where((Y_train == classes[1]))
testfilter0, testfilter1 = np.where((Y_test == classes[0])), np.where((Y_test == classes[1]))
X1_train, X0_train = X_train[trainfilter1], X_train[trainfilter0]
X1_test, X0_test = X_test[testfilter1], X_test[testfilter0]

#make new data
def new_data():
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

# Train the model
def train():
    loss_history = []
    model = Hybrid_nn.HybridModel().to(device)
    model.train()

    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    for it in range(iterations):
        X1_batch, X2_batch, Y_batch = new_data()
        X1_batch, X2_batch, Y_batch = X1_batch.to(device), X2_batch.to(device), Y_batch.to(device)

        pred = model(X1_batch, X2_batch)
        pred, y = pred.to(torch.float32), y.to(torch.float32)
        loss = loss_fn(pred, y)
        loss_history.append(loss.item())
        loss.requires_grad = True

        opt.zero_grad()
        loss.backward()
        opt.step()

        if it % 10 == 0:
            print(f"Iterations: {it} Loss: {loss.item()}")
    return loss_history


def train_distance():
    loss_history = []
    model = Hybrid_nn.HybridModel_distance().to(device)
    model.train()

    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    for it in range(iterations):
        
        batch_index = np.random.randint(0, len(min(len(X1_train), len(X0_train))), (batch_size,))
        X1_batch, X0_batch = X1_train[batch_index], X0_train[batch_index]
        X1_batch, X0_batch = torch.tensor(X1_batch).to(device), torch.tensor(X0_batch).to(device)

        distance = model(X1_batch, X0_batch)
        loss = -1 * distance
        loss_history.append(loss.item())

        opt.zero_grad()
        loss.backward()
        opt.step()

        if it % 10 == 0:
            print(f"Iterations: {it} Loss: {loss.item()}")

train()