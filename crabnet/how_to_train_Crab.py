import os
import numpy as np
import pandas as pd
import torch

from model import Model
from sklearn.metrics import roc_auc_score
from kingcrab import CrabNet
from get_compute_device import get_compute_device
from model_mlp import MLP
from kingcrab import combined_models

compute_device = get_compute_device(prefer_last=True)


# Create and load the weight for already existed model 

trained_crab_model =  Model(CrabNet(compute_device=compute_device).to(compute_device))
path = f'models/trained_models/trained_models_12_04_mat2vec/CritExam__Ed.pth'
network = torch.load(path, map_location=compute_device)
trained_crab_model.model.load_state_dict(network['weights'])

train_data = f'data/aflow__agl_thermal_conductivity_300K/train.csv'
val_data = f'data/aflow__agl_thermal_conductivity_300K/val.csv'
data_size = pd.read_csv(train_data).shape[0]
batch_size = 8
trained_crab_model.load_data(train_data, batch_size=batch_size, train=True)
trained_crab_model.load_data(val_data, batch_size=batch_size)
#trained_crab_model.fit(epochs=40, losscurve=False)


## Combination of my model + Crabnet 

new_model = Model(combined_models(pretrained_model=trained_crab_model.model,MLP=MLP(128,10,0,0,0,1)).to(compute_device))
new_model.load_data(train_data, batch_size=batch_size, train=True)
new_model.load_data(val_data, batch_size=batch_size)
new_model.fit(epochs=40, losscurve=False)
