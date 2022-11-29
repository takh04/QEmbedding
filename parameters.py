import torch

model = 'model1'
measure = 'Fidelity'
distance_measure = 'Trace'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parameters():
    return model, measure, distance_measure, device