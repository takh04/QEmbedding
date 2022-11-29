import torch

model = 'model1'
measure = 'Fidelity'
distance_measure = 'Trace'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_layers = 3

def get_parameters():
    return model, measure, distance_measure, device, N_layers