import data
import pennylane as qml
from pennylane import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt
import embedding
import parameters

model, measure, distance_measure, device, N_layers = parameters.get_parameters()
print(f"Uisng Device: {device}\n")

dev = qml.device('default.qubit', wires=8)
@qml.qnode(dev, interface="torch")
def distance_circuit1(inputs): 
  embedding.QuantumEmbedding1(inputs[0:8])
  return qml.density_matrix(wires=range(8))

@qml.qnode(dev, interface="torch")
def distance_circuit2(inputs):
  embedding.QuantumEmbedding2(inputs[0:16])
  return qml.density_matrix(wires=range(8))

class HybridModel_distance(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qlayer1 = qml.qnn.TorchLayer(distance_circuit1, weight_shapes={})
        self.qlayer2 = qml.qnn.TorchLayer(distance_circuit2, weight_shapes={})
        self.linear_relu_stack1 = nn.Sequential(
            nn.Linear(8, 10),
            nn.ReLU(),
            nn.Linear(10,10),
            nn.ReLU(),
            nn.Linear(10,8),
            nn.Softmax()
        )
        self.linear_relu_stack2 = nn.Sequential(
            nn.Linear(8, 20),
            nn.ReLU(),
            nn.Linear(20,20),
            nn.ReLU(),
            nn.Linear(20,16),
            nn.Softmax()
        )

        def forward(self, x1, x2):
            
            if model == 'model1':
                x1 = self.linear_relu_stack1(x1)
                x2 = self.linear_relu_stack1(x2)
                rhos1 = self.qlayer1(x1)
                rhos2 = self.qlayer1(x2)
            elif model == 'model2':
                x1 = self.linear_relu_stack2(x1)
                x2 = self.linear_relu_stack2(x2)
                rhos1 = self.qlayer2(x1)
                rhos2 = self.qlayer2(x2)


            rho1 = torch.sum(rhos1, dim=0) / len(x1)
            rho2 = torch.sum(rhos2, dim=0) / len(x2)
            rho_diff = rho1 - rho2

            if distance_measure == "Trace":
                eigvals = torch.linalg.eigvals(rho_diff)
                abs_eigvals = torch.abs(eigvals)
                return 0.5 * torch.real(torch.sum(abs_eigvals))
            elif distance_measure == "HS":
                return torch.trace(rho_diff @ rho_diff)
            elif distance_measure == "Fidelity":
                return torch.trace(torch.sqrtm(rho1) @ rho2 @ torch.sqrtm(rho1))

def model():
    return HybridModel_distance().to(device)