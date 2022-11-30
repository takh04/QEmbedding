import torch

model = 'model2'
measure = 'Fidelity'
distance_measure = 'Trace'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_layers = 3
