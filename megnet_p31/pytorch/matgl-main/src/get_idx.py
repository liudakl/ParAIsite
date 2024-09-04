#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 12:44:46 2024

@author: lklochko
"""


from __future__ import annotations
import os
import warnings
import torch 


from dgl.data.utils import split_dataset

from matgl.ext.pymatgen import Structure2Graph
from matgl.graph.data import MGLDataset, MGLDataLoader,collate_fn_graph

import numpy as np
import pickle 
from lightning.pytorch.callbacks import ModelCheckpoint
from matgl.config import DEFAULT_ELEMENTS

from custom_functions import return_dataset_train 


warnings.simplefilter("ignore")


# Setup dataset: 
    
dataset_name = 'L96'

SetToUse, structure = return_dataset_train (dataset_name)
thermal_conduct = SetToUse.TC.to_list()
elem_list = DEFAULT_ELEMENTS  #get_element_list(structure)
converter = Structure2Graph(element_types=elem_list, cutoff=4.0)



######     Model setup    ########

mp_dataset = MGLDataset(
    structures=structure,
    labels={"TC": thermal_conduct},
    converter=converter,
)
scaler = torch.load('/home/lklochko/Desktop/ProjPostDoc/GitHub/fine_tuning_p60/megnet_p31/pytorch/matgl-main/src/structures_scalers/torch.scaler')


maxRuns = 9

for nRuns in range (1,maxRuns+1):
    best_mape = np.inf
    checkpoint_callback = ModelCheckpoint(monitor='val_Total_Loss',dirpath='best_models/',filename='no_weights-%s_%s'%(dataset_name,nRuns))

    train_data, val_data = split_dataset(
    mp_dataset,
    frac_list=[0.8, 0.2],
    shuffle=True,
    random_state=nRuns,
)


    train_loader, val_loader = MGLDataLoader(
    train_data=train_data,
    val_data=val_data,
    collate_fn=collate_fn_graph,
    batch_size=8,
    num_workers=0,
)

    with open('best_models/val_idx_%s_%s.pkl'%(dataset_name,nRuns), 'wb') as f:
        pickle.dump(val_data.indices, f)
    
    with open('best_models/train_idx_%s_%s.pkl'%(dataset_name,nRuns), 'wb') as f:
        pickle.dump(train_data.indices, f)
        
        
    for fn in ("dgl_graph.bin", "lattice.pt", "dgl_line_graph.bin", "state_attr.pt", "labels.json"):
        try:
            os.remove(fn)
        except FileNotFoundError:
            pass


try:
    
    os.rename("/home/lklochko/Desktop/ProjPostDoc/GitHub/fine_tuning_p60/megnet_p31/pytorch/matgl-main/src/structures_scalers/torch.scaler", "/home/lklochko/Desktop/ProjPostDoc/GitHub/fine_tuning_p60/megnet_p31/pytorch/matgl-main/src/structures_scalers/torch.scaler.%s"%(dataset_name))
    os.remove("/home/lklochko/Desktop/ProjPostDoc/GitHub/fine_tuning_p60/megnet_p31/pytorch/matgl-main/src/structures_scalers/torch.scaler")
except FileNotFoundError:
    pass





