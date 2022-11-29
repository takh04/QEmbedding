import pennylane as qml
import parameters
from pennylane import numpy as np

N_layers = parameters.N_layers

def exp_Z(x, wires, inverse=False):
  if inverse == False:
    qml.RZ(-2 * x, wires=wires)
  elif inverse == True:
    qml.RZ(2 * x, wires=wires)

def exp_ZZ1(x, wires, inverse=False):
  if inverse == False:
    qml.CNOT(wires=wires)
    qml.RZ(-2 * x, wires=wires[1])
    qml.CNOT(wires=wires)
  elif inverse == True:
    qml.CNOT(wires=wires)
    qml.RZ(2 * x, wires=wires[1])
    qml.CNOT(wires=wires)

def exp_ZZ2(x1, x2, wires, inverse=False):
  if inverse == False:
    qml.CNOT(wires=wires)
    qml.RZ(-2 * (np.pi - x1) * (np.pi - x2), wires=wires[1])
    qml.CNOT(wires=wires)
  elif inverse == True:
    qml.CNOT(wires=wires)
    qml.RZ(2 * (np.pi - x1) * (np.pi - x2), wires=wires[1])
    qml.CNOT(wires=wires)



# Quantum Embedding 1 for model 1
def QuantumEmbedding1(input):
  for i in range(N_layers):
    for j in range(8):
      qml.Hadamard(wires=j)
      exp_Z(input[j], wires=j)
    for k in range(7):
      exp_ZZ2(input[k], input[k+1], wires=[k,k+1])
    exp_ZZ2(input[7], input[0], wires=[7, 0])

def QuantumEmbedding1_inverse(input):
  for i in range(N_layers):
    exp_ZZ2(input[7], input[0], wires=[7, 0], inverse=True)
    for k in reversed(range(7)):
      exp_ZZ2(input[k], input[k+1], wires=[k,k+1], inverse=True)
    qml.Barrier()
    for j in range(8):
      exp_Z(input[j], wires=j, inverse=True)
      qml.Hadamard(wires=j)

# Quantum Embedding 2 for model 2
def QuantumEmbedding2(input):
  for i in range(N_layers):
    for j in range(8):
      qml.Hadamard(wires=j)
      exp_Z(input[j], wires=j)
    for k in range(7):
      exp_ZZ1(input[8+k], wires=[k, k+1])
    exp_ZZ1(input[15], wires=[7,0])

def QuantumEmbedding2_inverse(input):
  for i in range(N_layers):
    exp_ZZ1(input[15], wires=[7,0], inverse=True)
    for k in reversed(range(7)):
      exp_ZZ1(input[k+8], wires=[k,k+1], inverse=True)
    qml.Barrier()
    for j in range(8):
      exp_Z(input[j], wires=j, inverse=True)
      qml.Hadamard(wires=j)