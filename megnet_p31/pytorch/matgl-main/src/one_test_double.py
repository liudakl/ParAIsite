#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:56:39 2024

@author: lklochko
"""
from __future__ import annotations
import warnings
import torch 
from matgl.utils.training import ModelLightningModule
import matgl 
from model_mlp import myMLP
import pickle 
import numpy as np
import pandas as pd
from matgl.models import combined_models
from custom_functions import create_changed_megned_model,unlog10, inverse_transform
from sklearn.metrics import mean_absolute_percentage_error


warnings.simplefilter("ignore")


model_name = 'MIX'
dataset_name = 'L96'


#### Load model ##### 

NN1 = 450
NN2 = 350
NN3 = 350
NN4 = 0
megnet_loaded = matgl.load_model("/Users/liudmylaklochko/Desktop/FineTuningRemote/fine_tuning_papers/megnet_p31/pytorch/matgl-main/pretrained_models/MEGNet-MP-2018.6.1-Eform")
model_megned_changed =  create_changed_megned_model() 
model_megned_changed.load_state_dict(megnet_loaded.state_dict(),strict=False)
mod_mlp = myMLP (16,NN1,NN2,NN3,NN4,1)
new_model = combined_models(pretrained_model=model_megned_changed,myMLP=mod_mlp)

checkpoint_path = 'best_models/double_train_AFLOW_on_%s_1.ckpt'%(model_name)

checkpoint = torch.load(checkpoint_path)
lit_module_loaded = ModelLightningModule(model=new_model,loss=checkpoint['hyper_parameters']['loss'], lr=checkpoint['hyper_parameters']['lr'], scaler=checkpoint['hyper_parameters']['scaler'])
lit_module_loaded.load_state_dict(checkpoint['state_dict'])
model_best = lit_module_loaded.model
model_best.train(False)




with open('best_models/val_idx_%s_1.pkl'%(dataset_name), 'rb') as f:
    val_idx = pickle.load(f)    


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


scaler = torch.load('structures_scalers/torch.scaler.%s'%(model_name))
with open ('structures_scalers/structures_%s.pkl'%(dataset_name), 'rb') as fp:
     structure = pickle.load(fp)    

y_pred = []
y_true = []

for idx in val_idx :
#for idx in  range(0,len(structure)):    
         preds = model_best.predict_structure(structure[idx])
         preds_ivT =  inverse_transform(scaler,preds)
         tc_pred = unlog10(preds_ivT).item()
         tc_true  = SetToUse.TC.iloc[idx]
         
         y_pred.append(tc_pred)
         y_true.append(tc_true)
     
y_pred = np.array(y_pred)
y_true = np.array(y_true)        
mape_run = mean_absolute_percentage_error(y_true,y_pred)
    
print("for model 1 mape is %0.3f ;"%(mape_run))




