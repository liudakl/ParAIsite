#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 13:14:57 2024

@author: lklochko
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
from matgl.models import MEGNet,combined_models,MEGNet_changed
from matgl.utils.io import RemoteFile
from matgl.utils.training import ModelLightningModule
import matgl 
from model_mlp import myMLP
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

train_data, val_data = split_dataset(
    mp_dataset,
    frac_list=[0.8, 0.2],
    shuffle=True,
    random_state=42,
)


train_loader, val_loader = MGLDataLoader(
    train_data=train_data,
    val_data=val_data,
    batch_size=8,
    num_workers=0,
)


node_embed = torch.nn.Embedding(len(elem_list), 16)
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


loaded_megnet = matgl.load_model("MEGNet-MP-2018.6.1-Eform")


model.load_state_dict(loaded_megnet.state_dict(),strict=False)
mod_mlp = myMLP (16,10,0,0,0,1)
new_model = combined_models(pretrained_model=model,myMLP=mod_mlp)

lit_module = ModelLightningModule(model=new_model,loss='l1_loss',lr=1e-3,scaler=scaler)

logger = CSVLogger("logs", name="MEGNet_training_%s"%(1),version=0)
trainer = pl.Trainer(max_epochs=100, accelerator="cpu", logger=logger)
trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
