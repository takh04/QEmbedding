import data
import torch
import parameters
import Hybrid_nn
import numpy as np

model, measure, distance_measure, device, N_layers = parameters.get_parameters()
batch_size = 50
print(f"Uisng Device: {device}\n")
feature_reduction = 'PCA8'
classes = [0,1]
X_train, X_test, Y_train, Y_test = data.data_load_and_process('mnist', '2', feature_reduction=feature_reduction, classes=classes)
loss_fn = torch.MSEloss()
optimizer = 1
iterations = 100

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
    return X1_new, X2_new, Y_new

def train():
    loss_history = []
    model.train()

    for it in range(iterations):
        X1_batch, X2_batch, Y_batch = new_data()
        X1_batch, X2_batch, Y_batch = X1_batch.to(device), X2_batch.to(device), Y_batch.to(device)

        pred = model(X1_batch, X2_batch)
        pred, y = pred.to(torch.float32), y.to(torch.float32)
        loss = loss_fn(pred, y)
        loss_history.append(loss.item())
        loss.requires_grad = True

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % 10 == 0:
            print(f"Iterations: {it} Loss: {loss.item()}")
    return loss_history


def train_distance():
    loss_history = []
    model.train()

    for it in range(iterations):
        
        X1_batch, X0_batch = X1_batch.to(device), X0_batch.to(device)

        distance = model(X1_batch, X0_batch)
        loss = -1 * distance
        loss_history.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % 10 == 0:
            print(f"Iterations: {it} Loss: {loss.item()}")

        