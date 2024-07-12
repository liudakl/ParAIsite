# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from __future__ import annotations
import pandas as pd 
from pymatgen.ext.matproj import MPRester
import os
import shutil
import warnings
import zipfile
import torch 

import matplotlib.pyplot as plt
import pandas as pd
import lightning as pl

from dgl.data.utils import split_dataset
from pymatgen.core import Structure
from pytorch_lightning.loggers import CSVLogger
from tqdm import tqdm

from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MGLDataset, MGLDataLoader, collate_fn_pes, collate_fn_graph
from matgl.layers import BondExpansion
from matgl.models import MEGNet,combined_models
from matgl.utils.io import RemoteFile
from matgl.utils.training import ModelLightningModule
import matgl 
from model_mlp import MLP
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle 



warnings.simplefilter("ignore")

SetToUse = pd.read_csv("https://gitlab.univ-lorraine.fr/klochko1/mdp/-/raw/main/cif_small_L.csv",index_col=0)
SetToUse.rename({'chemsys': 'formula','mpd_id':'ID','k_voigt':'kV', 'k_vrh':'kVRH', 'k_reuss':'kR','g_reuss':'gR','g_vrh':'gVRH','g_voigt':'gV'}, axis=1, inplace=True)
SetToUse = SetToUse.reset_index(drop=True)


with open ('structures_L96.pkl', 'rb') as fp:
    structure = pickle.load(fp)


# Setup dataset: 

thermal_conduct = SetToUse.TC.to_list()
elem_list = get_element_list(structure)
converter = Structure2Graph(element_types=elem_list, cutoff=4.0)


######     Model setup    ########
####   Load Pre trained model and combinde with the new one    ######

mp_dataset = MGLDataset(
    structures=structure,
    labels={"TC": thermal_conduct},
    converter=converter,
)

scaler = torch.load('/home/lklochko/Desktop/ProjPostDoc/GitHub/fine_tuning_p60/megnet_p31/pytorch/matgl-main/src/torch.scalerY')


best_mapes = [] 
maxRuns = 10

for nRuns in range (1,maxRuns+1):
    best_mape = np.inf
    
    train_data, val_data = split_dataset(
    mp_dataset,
    frac_list=[0.8, 0.2],
    shuffle=True,
    random_state=nRuns,
)


    train_loader, val_loader = MGLDataLoader(
    train_data=train_data,
    val_data=val_data,
    batch_size=8,
    num_workers=0,
)

 
    
    model_megnet = matgl.load_model("MEGNet-MP-2018.6.1-Eform")
    mod_mlp = MLP (160,350,350,350,350,1)
    new_model = combined_models(pretrained_model=model_megnet.model,MLP=mod_mlp)
    lit_module = ModelLightningModule(model=new_model,loss='l1_loss',lr=1e-3,scaler=scaler)

#####   Training   #######


    logger = CSVLogger("logs", name="MEGNet_training_%s"%(nRuns),version=0)
    trainer = pl.Trainer(max_epochs=100, accelerator="cpu", logger=logger)
    trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

#####   Plot Results  #######



    metrics = pd.read_csv("logs/MEGNet_training_%s/version_0/metrics.csv"%(nRuns))

    x1 = metrics["train_mape"].dropna().reset_index().drop(columns='index')
    x2 = metrics["val_mape"].dropna().reset_index().drop(columns='index')

    y = range(len(x1))

    plt.figure(figsize=(10, 5))
    plt.plot(y, x1,'-o', label='Train MAPE')
    plt.plot(y, x2, '-o', label='Validation MAPE')
    plt.xlabel('Epochs')
    plt.ylabel('MAPE')
    plt.title('run = %s'%(nRuns))
    plt.legend()
    
    plt.savefig("/home/lklochko/Desktop/ProjPostDoc/GitHub/fine_tuning_p60/megnet_p31/pytorch/matgl-main/results_plots/MAPE_for_run_%s.png"%(nRuns))
    plt.show()
    min_mape_val = x2.val_mape.min()
    
    if min_mape_val < best_mape:
        best_mape = min_mape_val
        best_mapes.append(best_mape)

    for fn in ("dgl_graph.bin", "lattice.pt", "dgl_line_graph.bin", "state_attr.pt", "labels.json"):
        try:
            os.remove(fn)
        except FileNotFoundError:
            pass

for nRuns in range (1,maxRuns+1): 
    shutil.rmtree("logs/MEGNet_training_%s"%(nRuns))

#shutil.rmtree("logs")

print("\n###############################\n")
print("best MAPE: %0.2f (%0.2f)\n"%(np.array(best_mapes).mean(),np.array(best_mapes).std()))
print("###############################")





