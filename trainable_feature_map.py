import pennylane as qml
from pennylane import numpy as np
import embedding
import data
import torch

def trainable_embedding(input, parameters):

    for i in range(4):
        qml.RY(parameters[i], wires=i)
    
    qml.RYY(parameters[4], wires=[0,1])
    qml.RYY(parameters[5], wires=[1,2])
    qml.RYY(parameters[6], wires=[2,3])
    qml.RYY(parameters[7], wires=[3,0])

    qml.RYY(parameters[8], wires=[0,2])
    qml.RYY(parameters[9], wires=[1,3])
    qml.RYY(parameters[10], wires=[2,0])
    qml.RYY(parameters[11], wires=[3,1])

    qml.RYY(parameters[12], wires=[0,3])
    qml.RYY(parameters[13], wires=[1,0])
    qml.RYY(parameters[14], wires=[2,1])
    qml.RYY(parameters[15], wires=[3,2])

    for i in range(4):
        qml.RX(parameters[i+16], wires=i)
    
    qml.RXX(parameters[20], wires=[0,1])
    qml.RXX(parameters[21], wires=[1,2])
    qml.RXX(parameters[22], wires=[2,3])
    qml.RXX(parameters[23], wires=[3,0])

    qml.RXX(parameters[24], wires=[0,2])
    qml.RXX(parameters[25], wires=[1,3])
    qml.RXX(parameters[26], wires=[2,0])
    qml.RXX(parameters[27], wires=[3,1])

    qml.RXX(parameters[28], wires=[0,3])
    qml.RXX(parameters[29], wires=[1,0])
    qml.RXX(parameters[30], wires=[2,1])
    qml.RXX(parameters[31], wires=[3,2])

    for j in range(4):
        qml.Hadamard(wires=j)
        embedding.exp_Z(input[j], wires=j)
    for k in range(3):
        embedding.exp_ZZ2(input[k], input[k+1], wires=[k,k+1])

dev = qml.device("default.qubit", wires=4)
@qml.qnode(dev, interface='torch')
def circuit(X1, X2, parameters):
    trainable_embedding(X1, parameters)
    qml.adjoint(trainable_embedding)(X2, parameters)
    return qml.probs(wires=range(4))

batch_size = 25
iterations = 1000
classes = [0, 1]
feature_reduction = 'PCA4'


#load data
X_train, X_test, Y_train, Y_test = data.data_load_and_process('mnist', feature_reduction=feature_reduction, classes=classes)
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

    return X1_new, X2_new, Y_new


def train_models():
    parameters = torch.randn(32, requries_grad=True)
    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.SGD(parameters, lr=0.01)
    for it in range(1000):
        X1_batch, X2_batch, Y_batch = new_data(batch_size, X_train, Y_train)
        pred = circuit(X1_batch, X2_batch)
        loss = loss_fn(pred, Y_batch)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
    
        if it % 10 == 0:
            print(f"Iterations: {it} Loss: {loss.item()}")
            