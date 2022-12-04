import pennylane as qml
import torch
from torch import nn
import embedding
import parameters

dev = qml.device('default.qubit', wires=8)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hybrid Model 1
@qml.qnode(dev, interface="torch")
def circuit1(inputs): 
    embedding.QuantumEmbedding1(inputs[0:8])
    embedding.QuantumEmbedding1_inverse(inputs[8:16])
    return qml.probs(wires=range(8))

class Model1_Fidelity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qlayer1 = qml.qnn.TorchLayer(circuit1, weight_shapes={})
        self.matrix_fn1 = qml.matrix(circuit1)
        self.linear_relu_stack1 = nn.Sequential(
            nn.Linear(8, 10),
            nn.ReLU(),
            nn.Linear(10,10),
            nn.ReLU(),
            nn.Linear(10,8)
        )
    def forward(self, x1, x2):
        x1 = self.linear_relu_stack1(x1)
        x2 = self.linear_relu_stack1(x2)
        x = torch.concat([x1, x2], 1)
        x = self.qlayer1(x)
        return x[:,0]


class Model1_HSinner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qlayer1 = qml.qnn.TorchLayer(circuit1, weight_shapes={})
        self.matrix_fn1 = qml.matrix(circuit1)
        self.linear_relu_stack1 = nn.Sequential(
            nn.Linear(8, 10),
            nn.ReLU(),
            nn.Linear(10,10),
            nn.ReLU(),
            nn.Linear(10,8)
        )
    def forward(self, x1, x2):
        x1 = self.linear_relu_stack1(x1)
        x2 = self.linear_relu_stack1(x2)
        x = torch.concat([x1, x2], 1).to("cpu")
        x = [torch.real(torch.trace(self.matrix_fn1(a))) for a in x]
        x = torch.stack(x, dim = 0).to(device)
        return x / 2**8



# Hybrid Model 2
@qml.qnode(dev, interface="torch")
def circuit2(inputs): 
    embedding.QuantumEmbedding2(inputs[0:16])
    embedding.QuantumEmbedding2_inverse(inputs[16:32])
    return qml.probs(wires=range(8))

class Model2_Fidelity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qlayer2 = qml.qnn.TorchLayer(circuit2, weight_shapes={})
        self.matrix_fn2 = qml.matrix(circuit2)
        self.linear_relu_stack2 = nn.Sequential(
            nn.Linear(8, 20),
            nn.ReLU(),
            nn.Linear(20,20),
            nn.ReLU(),
            nn.Linear(20,16)
        )
    def forward(self, x1, x2):
        x1 = self.linear_relu_stack2(x1)
        x2 = self.linear_relu_stack2(x2)
        x = torch.concat([x1, x2], 1)
        x = self.qlayer2(x)
        return x[:,0]

class Model2_HSinner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qlayer2 = qml.qnn.TorchLayer(circuit2, weight_shapes={})
        self.matrix_fn2 = qml.matrix(circuit2)
        self.linear_relu_stack2 = nn.Sequential(
            nn.Linear(8, 20),
            nn.ReLU(),
            nn.Linear(20,20),
            nn.ReLU(),
            nn.Linear(20,16)
        )
    def forward(self, x1, x2):
        x1 = self.linear_relu_stack2(x1)
        x2 = self.linear_relu_stack2(x2)
        x = torch.concat([x1, x2], 1).to("cpu")
        x = [torch.real(torch.trace(self.matrix_fn2(a))) for a in x]
        x = torch.stack(x, dim=0).to(device)
        return x / 2**8


# Hybrid Distance Model1
@qml.qnode(dev, interface="torch")
def distance_circuit1(inputs): 
    embedding.QuantumEmbedding1(inputs[0:8])
    return qml.density_matrix(wires=range(8))

class DistanceModel1_Trace(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qlayer1_distance = qml.qnn.TorchLayer(distance_circuit1, weight_shapes={})
        self.linear_relu_stack1 = nn.Sequential(
            nn.Linear(8, 10),
            nn.ReLU(),
            nn.Linear(10,10),
            nn.ReLU(),
            nn.Linear(10,8)
        )
    def forward(self, x1, x0):
        x1 = self.linear_relu_stack1(x1)
        x0 = self.linear_relu_stack1(x0)
        rhos1 = self.qlayer1_distance(x1)
        rhos0 = self.qlayer1_distance(x0)
        rho1 = torch.sum(rhos1, dim=0) / len(x1)
        rho0 = torch.sum(rhos0, dim=0) / len(x0)
        rho_diff = rho1 - rho0
        eigvals = torch.linalg.eigvals(rho_diff)
        return -0.5 * torch.real(torch.sum(torch.abs(eigvals)))


class DistanceModel1_HS(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qlayer1_distance = qml.qnn.TorchLayer(distance_circuit1, weight_shapes={})
        self.linear_relu_stack1 = nn.Sequential(
            nn.Linear(8, 10),
            nn.ReLU(),
            nn.Linear(10,10),
            nn.ReLU(),
            nn.Linear(10,8)
        )
    def forward(self, x1, x0):
        x1 = self.linear_relu_stack1(x1)
        x0 = self.linear_relu_stack1(x0)
        rhos1 = self.qlayer1_distance(x1)
        rhos0 = self.qlayer1_distance(x0)
        rho1 = torch.sum(rhos1, dim=0) / len(x1)
        rho0 = torch.sum(rhos0, dim=0) / len(x0)
        rho_diff = rho1 - rho0
        return -0.5 * torch.trace(rho_diff @ rho_diff)


# Hybrid Distance Model 2
@qml.qnode(dev, interface="torch")
def distance_circuit2(inputs):
    embedding.QuantumEmbedding2(inputs[0:16])
    return qml.density_matrix(wires=range(8))

class DistanceModel2_Trace(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qlayer2_distance = qml.qnn.TorchLayer(distance_circuit2, weight_shapes={})
        self.linear_relu_stack2 = nn.Sequential(
            nn.Linear(8, 20),
            nn.ReLU(),
            nn.Linear(20,20),
            nn.ReLU(),
            nn.Linear(20,16)
        )
    def forward(self, x1, x0):
        x1 = self.linear_relu_stack2(x1)
        x0 = self.linear_relu_stack2(x0)
        rhos1 = self.qlayer2_distance(x1)
        rhos0 = self.qlayer2_distance(x0)
        rho1 = torch.sum(rhos1, dim=0) / len(x1)
        rho0 = torch.sum(rhos0, dim=0) / len(x0)
        rho_diff = rho1 - rho0
        eigvals = torch.linalg.eigvals(rho_diff)
        return -0.5 * torch.real(torch.sum(torch.abs(eigvals)))


class DistanceModel2_HS(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qlayer2_distance = qml.qnn.TorchLayer(distance_circuit2, weight_shapes={})
        self.linear_relu_stack2 = nn.Sequential(
            nn.Linear(8, 20),
            nn.ReLU(),
            nn.Linear(20,20),
            nn.ReLU(),
            nn.Linear(20,16)
        )
    def forward(self, x1, x0):
        x1 = self.linear_relu_stack2(x1)
        x0 = self.linear_relu_stack2(x0)
        rhos1 = self.qlayer2_distance(x1)
        rhos0 = self.qlayer2_distance(x0)
        rho1 = torch.sum(rhos1, dim=0) / len(x1)
        rho0 = torch.sum(rhos0, dim=0) / len(x0)
        rho_diff = rho1 - rho0
        return -0.5 * torch.trace(rho_diff @ rho_diff)


# Get model function
def get_model(model):
    if model == 'Model1_Fidelity':
        return Model1_Fidelity()
    elif model == 'Model1_HSinner':
        return Model1_HSinner()
    elif model == 'Model2_Fidelity':
        return Model2_Fidelity()
    elif model == 'Model2_HSinner':
        return Model2_HSinner()
    elif model == 'DistanceModel1_Trace':
        return DistanceModel1_Trace()
    elif model == 'DistanceModel1_HS':
        return DistanceModel1_HS()
    elif model == 'DistanceModel2_Trace':
        return DistanceModel2_Trace()
    elif model == 'DistanceModel2_HS':
        return DistanceModel2_HS()