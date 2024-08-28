# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from __future__ import annotations
import pandas as pd 
import os
import shutil
import warnings
import torch 

import matplotlib.pyplot as plt
import lightning as pl

from dgl.data.utils import split_dataset
from pytorch_lightning.loggers import CSVLogger

from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MGLDataset, MGLDataLoader,collate_fn_graph
from matgl.models import combined_models
from matgl.utils.training import ModelLightningModule
import matgl 
from model_mlp import myMLP
import numpy as np
import pickle 
from lightning.pytorch.callbacks import ModelCheckpoint
from matgl.config import DEFAULT_ELEMENTS

from custom_functions import return_dataset_train,create_changed_megned_model 


warnings.simplefilter("ignore")


# Setup dataset: 
    
dataset_name = 'AFLOW'

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


best_mapes = [] 
maxRuns = 9
maxEpochs = 300
NN1 = 450
NN2 = 350
NN3 = 350
NN4 = 0


    


for nRuns in range (1,maxRuns+1):
    best_mape = np.inf
    checkpoint_callback = ModelCheckpoint(monitor='val_Total_Loss',dirpath='best_models/',filename='sample-%s_%s'%(dataset_name,nRuns))

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
    
    megnet_loaded = matgl.load_model("MEGNet-MP-2018.6.1-Eform")
    model_megned_changed =  create_changed_megned_model() 
    model_megned_changed.load_state_dict(megnet_loaded.state_dict(),strict=False)
    mod_mlp = myMLP (16,NN1,NN2,NN3,NN4,1)
    new_model = combined_models(pretrained_model=model_megned_changed,myMLP=mod_mlp)
    lit_module = ModelLightningModule(model=new_model,loss='l1_loss',lr=1e-3,scaler=scaler)

############   Training  Part   ############


    logger = CSVLogger("logs", name="MEGNet_m1_training_%s_%s"%(dataset_name,nRuns),version=0)
    trainer = pl.Trainer(max_epochs=maxEpochs, accelerator="cpu", logger=logger,callbacks=[checkpoint_callback])
    trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)



#####   MAPE Metrics Results  #######



    metrics = pd.read_csv("logs/MEGNet_m1_training_%s_%s/version_0/metrics.csv"%(dataset_name,nRuns))

    x1 = metrics["train_Total_Loss"].dropna().reset_index().drop(columns='index')
    x2 = metrics["val_Total_Loss"].dropna().reset_index().drop(columns='index')   
        
    min_mape_val = x2.val_Total_Loss.min()
    
    if min_mape_val < best_mape:
        best_mape = min_mape_val
        best_mapes.append(best_mape)      
        #torch.save(lit_module.model, "best_models/model_full_lit_%s."%(dataset_name)+str(nRuns)+".pt")

        
    

    for fn in ("dgl_graph.bin", "lattice.pt", "dgl_line_graph.bin", "state_attr.pt", "labels.json"):
        try:
            os.remove(fn)
        except FileNotFoundError:
            pass

#for nRuns in range (1,maxRuns+1): 
#    shutil.rmtree("logs/MEGNet_training_%s"%(nRuns))
try:
    
    os.rename("/home/lklochko/Desktop/ProjPostDoc/GitHub/fine_tuning_p60/megnet_p31/pytorch/matgl-main/src/structures_scalers/torch.scaler", "/home/lklochko/Desktop/ProjPostDoc/GitHub/fine_tuning_p60/megnet_p31/pytorch/matgl-main/src/structures_scalers/torch.scaler.%s"%(dataset_name))
    os.remove("/home/lklochko/Desktop/ProjPostDoc/GitHub/fine_tuning_p60/megnet_p31/pytorch/matgl-main/src/structures_scalers/torch.scaler")
except FileNotFoundError:
    pass


print("\n###############################")
print("#                             #")
print("#                             #")
print("#                             #")
print("#   best MAPE: %0.2f (%0.2f)    #"%(np.array(best_mapes).mean(),np.array(best_mapes).std()))
print("#                             #")
print("#                             #")
print("#                             #")
print("###############################")





