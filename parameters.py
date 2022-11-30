import torch

model = 'model1' #model1, model2, model1_distance, model2_distance
measure = 'Fidelity'     #fidelity, HS-inner, Hilber-Schmidt, Trace
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_layers = 3
