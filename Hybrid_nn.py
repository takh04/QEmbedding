import pennylane as qml
import torch
from torch import nn
import embedding
import parameters

model, measure, device = parameters.model, parameters.measure, parameters.device
dev = qml.device('default.qubit', wires=8)
dev2 = qml.device('default.qubit', wires=16)


# Hybrid Model 1
@qml.qnode(dev, interface="torch")
def circuit1(inputs): 
    embedding.QuantumEmbedding1(inputs[0:8])
    embedding.QuantumEmbedding1_inverse(inputs[8:16])
    return qml.probs(wires=range(8))

class HybridModel1(torch.nn.Module):
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

    if measure == 'Fidelity':
        def forward(self, x1, x2):
            x1 = self.linear_relu_stack1(x1)
            x2 = self.linear_relu_stack1(x2)
            x = torch.concat([x1, x2], 1)
            x = self.qlayer1(x)
            return x[:,0]
    elif measure == 'HS-inner':
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

class HybridModel2(torch.nn.Module):
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
    
    if measure == 'Fidelity':
        def forward(self, x1, x2):
            x1 = self.linear_relu_stack2(x1)
            x2 = self.linear_relu_stack2(x2)
            x = torch.concat([x1, x2], 1)
            x = self.qlayer2(x)
            return x[:,0]
    elif measure == 'HS-inner':
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

class HybridModel1_distance(torch.nn.Module):
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
        
        if measure == 'Trace':
            eigvals = torch.linalg.eigvals(rho_diff)
            return -0.5 * torch.real(torch.sum(torch.abs(eigvals)))
        elif measure == 'Hilbert-Schmidt':
            return -0.5 * torch.trace(rho_diff @ rho_diff)




# Hybrid Distance Model 2
@qml.qnode(dev, interface="torch")
def distance_circuit2(inputs):
    embedding.QuantumEmbedding2(inputs[0:16])
    return qml.density_matrix(wires=range(8))

class HybridModel2_distance(torch.nn.Module):
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

        if measure == 'Trace':
            eigvals = torch.linalg.eigvals(rho_diff)
            return -0.5 * torch.real(torch.sum(torch.abs(eigvals)))
        elif measure == 'Hilbert-Schmidt':
            return -0.5 * torch.trace(rho_diff @ rho_diff)

def get_model():
    if model == 'model1':
        return HybridModel1()
    elif model == 'model2':
        return HybridModel2()
    elif model == 'model1_distance':
        return HybridModel1_distance()
    elif model == 'model2_distance':
        return HybridModel2_distance()