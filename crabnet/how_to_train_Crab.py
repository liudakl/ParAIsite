import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from model import Model
from sklearn.metrics import roc_auc_score
from kingcrab import CrabNet
from get_compute_device import get_compute_device
from model_mlp import MLP
from kingcrab import combined_models

compute_device = get_compute_device(prefer_last=True)







path = f'models/trained_models/trained_models_12_04_mat2vec/aflow__agl_thermal_conductivity_300K.pth'

#train_data = f'data/aflow__agl_thermal_conductivity_300K/train.csv'
#val_data = f'data/aflow__agl_thermal_conductivity_300K/val.csv'
train_data = f'data/L96/train.csv'
val_data = f'data/L96/val.csv'



trained_crab_model =  Model(CrabNet(compute_device=compute_device).to(compute_device))
network = torch.load(path, map_location=compute_device)
trained_crab_model.model.load_state_dict(network['weights'])



batch_size = 8

## Combination of my model + Crabnet 

new_model = Model(combined_models(pretrained_model=trained_crab_model.model,MLP=MLP(128,10,0,0,0,1)).to(compute_device))




results = [] 

for ii in range(1,10):
    print("================================")
    df1 = pd.read_csv("https://gitlab.univ-lorraine.fr/klochko1/mdp/-/raw/main/cif_small_L.csv",index_col=0)
    df1.rename({'chemsys': 'formula','k_voigt':'kV', 'k_vrh':'kVRH', 'k_reuss':'kR','g_reuss':'gR','g_vrh':'gVRH','g_voigt':'gV'}, axis=1,inplace=True)
    SetToUse = df1 [['formula','TC']].copy()
    target   = "TC"
    SetToUse.rename({target: 'target'}, axis=1, inplace=True)
    train_df, val_df = train_test_split(SetToUse, test_size=0.2, random_state=ii,shuffle=True)
    train_df.to_csv('data/L96/train.csv',index=False,header=True )
    val_df.to_csv('data/L96/train.csv',index=False,header=True )
    new_model.load_data(train_data, batch_size=batch_size, train=True)
    new_model.load_data(val_data, batch_size=batch_size)
    res=new_model.fit(epochs=100, losscurve=False, nrun=ii)
    results.append(res)

