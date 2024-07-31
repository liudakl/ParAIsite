#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 11:13:34 2024

@author: lklochko
"""

from __future__ import annotations
import pandas as pd 
import warnings
import torch 
from matgl.layers import BondExpansion
from matgl.models import combined_models,MEGNet_changed
from matgl.utils.training import ModelLightningModule
import matgl 
from model_mlp import myMLP
import numpy as np
import pickle 
from sklearn.metrics import mean_absolute_percentage_error



warnings.simplefilter("ignore")

def unlog10(data):
        data_unlog10 = 10**(data)-1
        return data_unlog10
    
def inverse_transform(scalerY,X):
        if scalerY.mean is None or scalerY.std is None:
            raise ValueError("The StandardScaler has not been fitted yet.")
        return X * scalerY.std + scalerY.mean 
    
def create_changed_megned_model () :
    bond_expansion = BondExpansion(rbf_type="Gaussian", initial=0.0, final=5.0, num_centers=100, width=0.5)

    model = MEGNet_changed(
        dim_node_embedding=16,
        dim_edge_embedding=100,
        dim_state_embedding=2,
        nblocks=3,
        hidden_layer_sizes_input=(64, 32),
        hidden_layer_sizes_conv=(64, 64, 32),
        nlayers_set2set=1,
        niters_set2set=2,
        hidden_layer_sizes_output=(32, 16),
        is_classification=False,
        activation_type="softplus2",
        bond_expansion=bond_expansion,
        cutoff=4.0,
        gauss_width=0.5,
    )
    
    return model 


def return_dataset (dataset_name, model_name):
    
    
    with open ('structures_%s.pkl'%(dataset_name), 'rb') as fp:
        structure = pickle.load(fp)
        
    if dataset_name == 'L96':
        df1 = pd.read_csv("https://gitlab.univ-lorraine.fr/klochko1/mdp/-/raw/main/cif_small_L.csv",index_col=0)
        df1.rename({'chemsys': 'formula','k_voigt':'kV', 'k_vrh':'kVRH', 'k_reuss':'kR','g_reuss':'gR','g_vrh':'gVRH','g_voigt':'gV'}, axis=1,inplace=True)
        SetToUse = df1[['TC']].copy()
    elif dataset_name == 'HH143':
        df2 = pd.read_csv("https://gitlab.univ-lorraine.fr/klochko1/mdp/-/raw/main/hh_143.csv", delimiter=';')
        df2 = df2.reset_index(drop=True)
        SetToUse = df2[['TC']].copy()
    elif dataset_name == 'MIX':
        df1 = pd.read_csv("https://gitlab.univ-lorraine.fr/klochko1/mdp/-/raw/main/cif_small_L.csv",index_col=0)
        df1.rename({'chemsys': 'formula','k_voigt':'kV', 'k_vrh':'kVRH', 'k_reuss':'kR','g_reuss':'gR','g_vrh':'gVRH','g_voigt':'gV'}, axis=1,inplace=True)

        df2 = pd.read_csv("https://gitlab.univ-lorraine.fr/klochko1/mdp/-/raw/main/hh_143.csv", delimiter=';')
        df2 = df2.reset_index(drop=True)

        df_1 = df1[['mpd_id','TC']]
        df_2 = df2[['mpd_id','TC']]

        SetToUse = pd.concat([df_1,df_2], ignore_index=True)
   
    scalerY = torch.load('/home/lklochko/Desktop/ProjPostDoc/GitHub/fine_tuning_p60/megnet_p31/pytorch/matgl-main/src/torch.scaler.%s'%(model_name))
    
    
    return   SetToUse, structure,scalerY



def mape_run_model (model, scaler, structure,val_idx):
    y_pred = []
    y_true = []
    for idx in val_idx :#in range(0,len(structure)):
        preds = model.predict_structure(structure[idx])
        preds_ivT =  inverse_transform(scaler,preds)
        tc_pred = unlog10(preds_ivT).item()
        tc_true  = SetToUse.TC.iloc[idx]
            
        y_pred.append(tc_pred)
        y_true.append(tc_true)
        
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    mape_run = mean_absolute_percentage_error(y_true,y_pred)
    return mape_run

best_mapes = [] 
maxRuns = 1
nRuns = 1 
maxEpochs = 300 
NN1 = 450
NN2 = 350
NN3 = 350
NN4 = 0

dataset_name = 'L96'    
model_to_test      = 'L96'
SetToUse, structure, scaler = return_dataset (dataset_name,model_to_test)

with open('best_models/val_idx_2.pkl', 'rb') as f:
    val_idx = pickle.load(f)





##################  create a new model ###################### 
        
megnet_loaded = matgl.load_model("MEGNet-MP-2018.6.1-Eform")
model_megned_changed =  create_changed_megned_model() 
model_megned_changed.load_state_dict(megnet_loaded.state_dict(),strict=False)
mod_mlp = myMLP (16,NN1,NN2,NN3,NN4,1)
new_model = combined_models(pretrained_model=model_megned_changed,myMLP=mod_mlp)
checkpoint_path = 'best_models/sample-L96_2.ckpt'
checkpoint = torch.load(checkpoint_path)
lit_module_loaded = ModelLightningModule(model=new_model,loss=checkpoint['hyper_parameters']['loss'], lr=checkpoint['hyper_parameters']['lr'], scaler=checkpoint['hyper_parameters']['scaler'])
lit_module_loaded.load_state_dict(checkpoint['state_dict'])
model_best = lit_module_loaded.model
model_best.train(False)
mape_model_best =  mape_run_model (model_best, scaler, structure,val_idx)


model_lit_full = torch.load('best_models/model_full_lit_L96.2.pt')
mape_lit_full = mape_run_model (model_lit_full, scaler, structure,val_idx)


print("mape_model_best:",mape_model_best)
print("mape_lit_full",mape_lit_full)
