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





### ============  Create the original model Crabnet ============ ### 

path_to_model  = f'models/trained_models/trained_models_12_04_mat2vec/aflow__agl_thermal_conductivity_300K.pth'
trained_crab_model =  Model(CrabNet(compute_device=compute_device).to(compute_device))
network = torch.load(path_to_model, map_location=compute_device)
trained_crab_model.model.load_state_dict(network['weights'])


### ============  Create combined model: MLP + Crabnet ============ ###

new_model = Model(combined_models(pretrained_model=trained_crab_model.model,MLP=MLP(128,10,0,0,0,1)).to(compute_device))


 ### ============ Data preparation part  ============ ###


batch_size = 8
dataset    = 'L96' 

if dataset == 'L96':
    
    df1 = pd.read_csv("https://gitlab.univ-lorraine.fr/klochko1/mdp/-/raw/main/cif_small_L.csv",index_col=0)
    df1.rename({'chemsys': 'formula','k_voigt':'kV', 'k_vrh':'kVRH', 'k_reuss':'kR','g_reuss':'gR','g_vrh':'gVRH','g_voigt':'gV'}, axis=1,inplace=True)
    SetToUse = df1 [['formula','TC']].copy()
    target   = "TC"
    SetToUse.rename({target: 'target'}, axis=1, inplace=True)
    
    
elif dataset == 'mix_143':
    df2 = pd.read_csv("https://gitlab.univ-lorraine.fr/klochko1/mdp/-/raw/main/hh_143.csv", delimiter=';')
    df1 = pd.read_csv("https://gitlab.univ-lorraine.fr/klochko1/mdp/-/raw/main/cif_small_L.csv",index_col=0)
    df1.rename({'chemsys': 'formula','k_voigt':'kV', 'k_vrh':'kVRH', 'k_reuss':'kR','g_reuss':'gR','g_vrh':'gVRH','g_voigt':'gV'}, axis=1,inplace=True)
    df_1 = df1[['formula','TC']].copy()
    df_2 = df2[['formula','TC']].copy()
    SetToUse = pd.concat([df_1,df_2], ignore_index=True)
    target   = "TC"
    SetToUse.rename({target: 'target'}, axis=1, inplace=True)

else: 
    train_data = f'data/aflow__agl_thermal_conductivity_300K/train.csv'
    val_data = f'data/aflow__agl_thermal_conductivity_300K/val.csv'

data_path = f'data/aflow__agl_thermal_conductivity_300K'
train_data = data_path + "/train.csv"
val_data   = data_path + "/val.csv"


#train_data = f'data/aflow__agl_thermal_conductivity_300K/train.csv'
#val_data = f'data/aflow__agl_thermal_conductivity_300K/val.csv'
#train_data = f'data/L96/train.csv'
#val_data = f'data/L96/val.csv'



# Executing learing part: 

epochs = 500



results = [] 

for ii in range(1,10):
    print("================================")
    
    if dataset == 'L96' or dataset == 'mix_143' :
        train_df, val_df = train_test_split(SetToUse, test_size=0.2, random_state=ii,shuffle=True)
        train_df.to_csv('%s'%(train_data),index=False,header=True )
        val_df.to_csv('%s'%(val_data),index=False,header=True )

    new_model.load_data(train_data, batch_size=batch_size, train=True)
    new_model.load_data(val_data, batch_size=batch_size)
    res=new_model.fit(epochs=epochs, losscurve=False, nrun=ii)
    results.append(res)


# Printing output results

data = [item for sublist in results for item in sublist if item is not None]
df = pd.DataFrame(data)
bests_mape_v = df.groupby("run").mape_v.min()
bests_mae_v = df.groupby("run").mae_v.min()

print("=======================================\n")
print("Best MAPE on validation over %s epochs: "%(epochs),round(bests_mape_v.mean(), 2), "(%s)"%(round(bests_mape_v.std(), 2)) )
print("Best MAE on validation over %s epochs: "%(epochs),round(bests_mae_v.mean(), 2), "(%s)"%(round(bests_mae_v.std(), 2)) )