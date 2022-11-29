import torch

model = 'model1'
measure = 'HS-inner'
distance_measure = 'Trace'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_layers = 3
