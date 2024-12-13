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
import sys
from matgl.config import DEFAULT_ELEMENTS
from matgl.ext.pymatgen import Structure2Graph
from matgl.graph.data import MGLDataset, MGLDataLoader,collate_fn_graph, MGLDataLoader_multiple
import os 

def setup_dataset_to_test(dataset_name_test):
    elem_list = DEFAULT_ELEMENTS  #get_element_list(structure)
    converter = Structure2Graph(element_types=elem_list, cutoff=4.0)
    if dataset_name_test == 'MIX':
        dataset_1 = 'Dataset1'
        dataset_2 = 'Dataset2'
        SetToUse_1, structure_1 = return_dataset_train (dataset_1)
        thermal_conduct = SetToUse_1.TC.to_list()
        
        mp_dataset_1 = MGLDataset(
            structures=structure_1,
            labels={"TC": thermal_conduct},
            converter=converter,
        )
        
        SetToUse_2, structure_2 = return_dataset_train (dataset_2)
        thermal_conduct = SetToUse_2.TC.to_list()
        
        mp_dataset_2 = MGLDataset(
            structures=structure_2,
            labels={"TC": thermal_conduct},
            converter=converter,
        )
        
        try:
            os.remove("structures_scalers/torch.scaler")
        except FileNotFoundError:
            pass
        
        SetToUse_mix, structure_mix = return_dataset_train (dataset_name_test)
        thermal_conduct = SetToUse_mix.TC.to_list()
        
        mp_dataset = MGLDataset(
            structures=structure_mix,
            labels={"TC": thermal_conduct},
            converter=converter,
        )
    
        return [], mp_dataset, mp_dataset_1, mp_dataset_2
    else:
        SetToUse_test, structure_test = return_dataset_train (dataset_name_test)
        thermal_conduct = SetToUse_test.TC.to_list()
    
        mp_dataset_test = MGLDataset(
        structures=structure_test,
        labels={"TC": thermal_conduct},
        converter=converter,
        )
        return [],  mp_dataset_test
    


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
    if dataset_name not in ['Dataset1','Dataset2','MIX','AFLOW']:
        raise ValueError("Data has not been loaded. Chose the dataset first. If you want to create a custom dataset, please follow README.")
        sys.exit(0)
    else:
       with open ('structures_scalers/structures_%s.pkl'%(dataset_name), 'rb') as fp:
           structure = pickle.load(fp)        
       with open ('structures_scalers/%s.pkl'%(dataset_name), 'rb') as fp:
            SetToUse = pickle.load(fp)        
        
    return   SetToUse, structure




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

def welcome():
    print ()
    print(" _____                  _____      __       ", "  ,---.")
    print("|  __ \\           /\\   |_   _|   (_) |      ", " ( @ @ )")
    print("| |__) |_ _ _ __ /  \\    | |  ___ _| |_ ___ ", "  ).-.(")
    print("|  ___/ _` | '__/ /\ \\   | | / __| | __/ _ \\", " '/|||\`")
    print("| |  | (_| | | / ____ \\ _| |_\__ \\ | ||  __/", "   '|`")
    print("|_|   \\__,_|_|/_/    \\_\\_____|___/_|\\__\\___|")

    print()
