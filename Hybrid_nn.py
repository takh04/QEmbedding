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
def circuit1(inputs): # circuit1 for model 1
  embedding.QuantumEmbedding1(inputs[0:8])
  embedding.QuantumEmbedding1_inverse(inputs[8:16])
  return qml.probs(wires=range(8))

@qml.qnode(dev, interface="torch")
def circuit2(inputs): # circuit2 for model 2
  embedding.QuantumEmbedding2(inputs[0:16])
  embedding.QuantumEmbedding2_inverse(inputs[16:32])
  return qml.probs(wires=range(8))

class HybridModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qlayer1 = qml.qnn.TorchLayer(circuit1, weight_shapes={})
        self.qlayer2 = qml.qnn.TorchLayer(circuit2, weight_shapes={})
        self.matrix_fn1 = qml.matrix(circuit1)
        self.matrix_fn2 = qml.matrix(circuit2)
        self.linear_relu_stack = nn.Sequential(
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

    if model == "model1":
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
                x = torch.concat([x1, x2], 1)
                x = [torch.real(torch.trace(self.matrix_fn1(a))) for a in x.to("cpu")]
                x = torch.tensor(x).to(device)
                return x / 2**8
    
    elif model == 'model2':
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
                x = torch.concat([x1, x2], 1)
                x = [torch.real(torch.trace(self.matrix_fn2(a))) for a in x.to("cpu")]
                x = torch.tensor(x).to(device)
                return x / 2**8

  
def hybrid_model():
    return HybridModel().to(device)