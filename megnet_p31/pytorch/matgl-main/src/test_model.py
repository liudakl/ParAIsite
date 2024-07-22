#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:56:39 2024

@author: lklochko
"""
from __future__ import annotations
import pandas as pd 
import numpy as np 
import torch 
import pickle 
from sklearn.metrics import mean_absolute_percentage_error

def unlog10(data):
        data_unlog10 = 10**(data)-1
        return data_unlog10
    
def inverse_transform(scalerY,X):
        if scalerY.mean is None or scalerY.std is None:
            raise ValueError("The StandardScaler has not been fitted yet.")
        return X * scalerY.std + scalerY.mean    


def return_dataset (dataset_name):
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
    scalerY = torch.load('/home/lklochko/Desktop/ProjPostDoc/GitHub/fine_tuning_p60/megnet_p31/pytorch/matgl-main/src/torch.scaler.%s'%(dataset_name))
    
    
    return   SetToUse, structure,scalerY
    

dataset_name = 'L96'    
test_on      = 'HH143'

nRunsmax = 9

SetToUse, structure, scalerY = return_dataset (dataset_name)


print("Test DataSet %s on %s pre-trained models on %s"%(dataset_name,nRunsmax,test_on))
    
mapes_all = []
for nRuns in range (1,nRunsmax+1):
    model = torch.load('best_models/model_%s.%s'%(test_on,nRuns))
    model.train(False)
    y_pred = []
    y_true = []
    for idx in range(0,len(structure)):
        preds = model.predict_structure(structure[idx])
        preds_ivT =  inverse_transform(scalerY,preds)
        tc_pred = unlog10(preds_ivT).item()
        tc_true  = SetToUse.TC.iloc[idx]
        
        y_pred.append(tc_pred)
        y_true.append(tc_true)
    
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    mape_run = mean_absolute_percentage_error(y_true,y_pred)
    mapes_all.append(mape_run)
    print("for model %s mape is %0.3f ;"%(nRuns,mape_run))

print("\n###############################")
print("#                             #")
print("#                             #")
print("#                             #")
print("#between models MAPE: %0.2f (%0.2f)    #"%(np.array(mapes_all).mean(),np.array(mapes_all).std()))
print("#                             #")
print("#                             #")
print("#                             #")
print("###############################")


