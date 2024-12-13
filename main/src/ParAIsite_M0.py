# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from __future__ import annotations
import pandas as pd 
import os
import warnings
import torch 

import lightning as pl

from dgl.data.utils import split_dataset
from pytorch_lightning.loggers import CSVLogger

from matgl.ext.pymatgen import Structure2Graph
from matgl.graph.data import MGLDataset, MGLDataLoader,collate_fn_graph, MGLDataLoader_multiple
from matgl.models import combined_models
from matgl.utils.training import ModelLightningModule
from model_mlp import myMLP
import numpy as np
from lightning.pytorch.callbacks import ModelCheckpoint
from matgl.config import DEFAULT_ELEMENTS

from custom_functions import return_dataset_train,create_changed_megned_model 


warnings.simplefilter("ignore")
elem_list = DEFAULT_ELEMENTS  #get_element_list(structure)
converter = Structure2Graph(element_types=elem_list, cutoff=4.0)


# ===================================================# 

# Setup dataset to test: 
  
dataset_name_test1 = 'Dataset2'
res_tes1_Dataset2 = [] 

SetToUse_test1, structure_test1 = return_dataset_train (dataset_name_test1)
thermal_conduct = SetToUse_test1.TC.to_list()
elem_list = DEFAULT_ELEMENTS  #get_element_list(structure)
converter_test = Structure2Graph(element_types=elem_list, cutoff=4.0)

mp_dataset_test1 = MGLDataset(
    structures=structure_test1,
    labels={"TC": thermal_conduct},
    converter=converter_test,
)

dataset_name_test2 = 'Dataset1'
res_tes2_Dataset1 = [] 

SetToUse_test2, structure_test2 = return_dataset_train (dataset_name_test2)
thermal_conduct = SetToUse_test2.TC.to_list()
elem_list = DEFAULT_ELEMENTS  #get_element_list(structure)
converter_test = Structure2Graph(element_types=elem_list, cutoff=4.0)

mp_dataset_test2 = MGLDataset(
    structures=structure_test2,
    labels={"TC": thermal_conduct},
    converter=converter_test,
)


res_tes3_MIX = [] 


dataset_name_test4 = 'AFLOW'
res_tes4_AFLOW = [] 


SetToUse_test4, structure_test4 = return_dataset_train (dataset_name_test4)
thermal_conduct = SetToUse_test4.TC.to_list()
elem_list = DEFAULT_ELEMENTS  #get_element_list(structure)
converter_test = Structure2Graph(element_types=elem_list, cutoff=4.0)

mp_dataset_test4 = MGLDataset(
    structures=structure_test4,
    labels={"TC": thermal_conduct},
    converter=converter_test,
)


try:
    
    os.remove("structures_scalers/torch.scaler")
except FileNotFoundError:
    pass


# ===================================================# 
 

# Setup dataset to TRAIN : 
    
dataset_name_TRAIN = 'Dataset1'


if dataset_name_TRAIN == 'MIX':
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
    
    SetToUse_mix, structure_mix = return_dataset_train (dataset_name_TRAIN)
    thermal_conduct = SetToUse_mix.TC.to_list()
    
    mp_dataset = MGLDataset(
        structures=structure_mix,
        labels={"TC": thermal_conduct},
        converter=converter,
    )


else:
    SetToUse, structure = return_dataset_train (dataset_name_TRAIN)
    thermal_conduct = SetToUse.TC.to_list()
    
    mp_dataset = MGLDataset(
        structures=structure,
        labels={"TC": thermal_conduct},
        converter=converter,
    )
    

scaler = torch.load('structures_scalers/torch.scaler')





######     Model setup    ########




best_mapes = [] 
maxRuns = 12
maxEpochs = 10
NN1 = 450
NN2 = 350
NN3 = 350
NN4 = 0
torchseed = 42 
device = "cuda" if torch.cuda.is_available() else "cpu"
print("RUNNIN ON", device)

pl.seed_everything(torchseed, workers=True)
torch.manual_seed(torchseed)
torch.cuda.manual_seed(torchseed)
    


for nRuns in range (11,maxRuns+1):
    best_mape = np.inf
    checkpoint_callback = ModelCheckpoint(monitor='val_Total_Loss',dirpath='best_models/',filename='no_weights-%s_%s'%(dataset_name_TRAIN,nRuns))
    
    if dataset_name_TRAIN == 'MIX': 
        
        train_data_1, val_data_1 = split_dataset(
        mp_dataset_1,
        frac_list=[0.8, 0.2],
        shuffle=True,
        random_state=nRuns,
    )
   
        train_data_2, val_data_2 = split_dataset(
        mp_dataset_2,
        frac_list=[0.8, 0.2],
        shuffle=True,
        random_state=nRuns,
    )

        train_loader, val_loader = MGLDataLoader_multiple(
              train_data_1=train_data_1,
              val_data_1=val_data_1,
              train_data_2=train_data_2,
              val_data_2=val_data_2,
              collate_fn=collate_fn_graph,
              batch_size=8,
              num_workers=0,
          )

    
    else: 
        train_data, val_data = split_dataset(
            mp_dataset,
            frac_list=[0.8, 0.2],
            shuffle=True,
            random_state=nRuns)

        train_loader, val_loader = MGLDataLoader(
            train_data=train_data,
            val_data=val_data,
            collate_fn=collate_fn_graph,
            batch_size=8,
            num_workers=0 )
    
    model_megned_changed =  create_changed_megned_model() 
    mod_mlp = myMLP (16,NN1,NN2,NN3,NN4,1).to(device)
    new_model = combined_models(pretrained_model=model_megned_changed,myMLP=mod_mlp).to(device)
    lit_module = ModelLightningModule(model=new_model,loss='l1_loss',lr=1e-3,scaler=scaler).to(device)

############   Training  Part   ############


    logger = CSVLogger("logs", name="MEGNet_training_no_weights_%s_%s"%(dataset_name_TRAIN,nRuns),version=0)
    trainer = pl.Trainer(max_epochs=maxEpochs,devices="auto", accelerator="gpu", logger=logger,callbacks=[checkpoint_callback])
    trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

# ===================================================# 

    train_data_test1, val_data_test1 = split_dataset(
    mp_dataset_test1,
    frac_list=[0.8, 0.2],
    shuffle=True,
    random_state=nRuns,
)


    train_loader_test1, val_loader_test1 = MGLDataLoader(
    train_data=train_data_test1,
    val_data=val_data_test1,
    collate_fn=collate_fn_graph,
    batch_size=8,
    num_workers=0,
)

    train_data_test2, val_data_test2 = split_dataset(
    mp_dataset_test2,
    frac_list=[0.8, 0.2],
    shuffle=True,
    random_state=nRuns,
)


    train_loader_test2, val_loader_test2 = MGLDataLoader(
    train_data=train_data_test2,
    val_data=val_data_test2,
    collate_fn=collate_fn_graph,
    batch_size=8,
    num_workers=0,
)


    train_loader_test3, val_loader_test3 = MGLDataLoader_multiple(
          train_data_1=train_data_test1,
          val_data_1=val_data_test1,
          train_data_2=train_data_test2,
          val_data_2=val_data_test2,
          collate_fn=collate_fn_graph,
          batch_size=8,
          num_workers=0,
      )


    train_data_test4, val_data_test4 = split_dataset(
    mp_dataset_test4,
    frac_list=[0.8, 0.2],
    shuffle=True,
    random_state=nRuns,
)


    train_loader_test4, val_loader_test4 = MGLDataLoader(
    train_data=train_data_test4,
    val_data=val_data_test4,
    collate_fn=collate_fn_graph,
    batch_size=8,
    num_workers=0,
)  
    
    res_test1 = trainer.test(dataloaders=val_loader_test1)
    res_test2 = trainer.test(dataloaders=val_loader_test2)
    res_test3 = trainer.test(dataloaders=val_loader_test3)
    res_test4 = trainer.test(dataloaders=val_loader_test4)
    
    
    res_tes1_Dataset2.append(list(res_test1[0].values())[0])
    res_tes2_Dataset1.append(list(res_test2[0].values())[0])
    res_tes3_MIX.append(list(res_test3[0].values())[0])
    res_tes4_AFLOW.append(list(res_test4[0].values())[0])

#####   MAPE Metrics Results  #######



    metrics = pd.read_csv("logs/MEGNet_training_no_weights_%s_%s/version_0/metrics.csv"%(dataset_name_TRAIN,nRuns))

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
    
    os.rename("structures_scalers/torch.scaler", "structures_scalers/torch.scaler.%s"%(dataset_name_TRAIN))
    os.remove("structures_scalers/torch.scaler")
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


df_final  = pd.DataFrame({
    'Run': range(1, maxRuns + 1),
    'train_on': '%s'%(dataset_name_TRAIN),
    'test_Dataset2': res_tes1_Dataset2,
    'test_Dataset1': res_tes2_Dataset1,
    'test_MIX': res_tes3_MIX,
    'test_AFLOW': res_tes4_AFLOW
})

df_final.to_csv('results_on_train_test/results_M0_trained_on_%s.csv'%(dataset_name_TRAIN), index=False)
