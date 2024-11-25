#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import pandas as pd 
import torch 
import numpy as np
import pickle 
from sklearn.metrics import mean_absolute_percentage_error
from matgl.layers import BondExpansion
from matgl.models import MEGNet_changed


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



def unlog10(data):
        data_unlog10 = 10**(data)-1
        return data_unlog10
    
def inverse_transform(scalerY,X):
        if scalerY.mean is None or scalerY.std is None:
            raise ValueError("The StandardScaler has not been fitted yet.")
        return X * scalerY.std + scalerY.mean    
    
    
def return_dataset_train (dataset_name):
    
    
    with open ('structures_scalers/structures_%s.pkl'%(dataset_name), 'rb') as fp:
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
    elif dataset_name == 'AFLOW':
        with open('structures_scalers/TC_AFLOW.pkl', 'rb') as fp:
            thermalCond  = pickle.load(fp)
            SetToUse = pd.DataFrame(np.array(thermalCond),columns=['TC']) 
        
    else:
        raise ValueError("Data has not been loaded. Chose dataset first.")

    
    
    return   SetToUse, structure


def return_dataset_test (dataset_name, model_name):
        
    with open ('structures_scalers/structures_%s.pkl'%(dataset_name), 'rb') as fp:
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

    elif dataset_name == 'AFLOW':
        with open('structures_scalers/TC_AFLOW.pkl', 'rb') as fp:
            thermalCond  = pickle.load(fp)
            SetToUse = pd.DataFrame(np.array(thermalCond),columns=['TC']) 
    
    else:
        raise ValueError("Data has not been loaded. Chose dataset first.")    
    scalerY = torch.load('structures_scalers/torch.scaler.%s'%(model_name))
    
    
    return   SetToUse, structure, scalerY


def mape_run_model (SetToUse, model, scaler, structure, val_idx, full_set=None):
    y_pred = []
    y_true = []
    if full_set == None or full_set == False :        
        for idx in val_idx :#in range(0,len(structure)):
            preds = model.predict_structure(structure[idx])
            if ~preds.isnan():
                preds_ivT =  inverse_transform(scaler,preds)
                tc_pred = unlog10(preds_ivT).item()
                tc_true  = SetToUse.TC.iloc[idx]
            
                y_pred.append(tc_pred)
                y_true.append(tc_true)        
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        
    elif full_set == True:
        for idx in  range(0,len(structure)):
            preds = model.predict_structure(structure[idx])
            if ~preds.isnan():
                preds_ivT =  inverse_transform(scaler,preds)
                tc_pred = unlog10(preds_ivT).item()
                tc_true  = SetToUse.TC.iloc[idx]
            
                y_pred.append(tc_pred)
                y_true.append(tc_true)
        
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)        
    mape_run = mean_absolute_percentage_error(y_true,y_pred)
    return mape_run
