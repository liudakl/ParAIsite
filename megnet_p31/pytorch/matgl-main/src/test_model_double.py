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
import numpy as np
import pickle 
from matgl.models import combined_models
from custom_functions import return_dataset_test,create_changed_megned_model,mape_run_model


warnings.simplefilter("ignore")


def restore_model (model_to_test,nRuns):
    NN1 = 450
    NN2 = 350
    NN3 = 350
    NN4 = 0
    megnet_loaded = matgl.load_model("MEGNet-MP-2018.6.1-Eform")
    model_megned_changed =  create_changed_megned_model() 
    model_megned_changed.load_state_dict(megnet_loaded.state_dict(),strict=False)
    mod_mlp = myMLP (16,NN1,NN2,NN3,NN4,1)
    new_model = combined_models(pretrained_model=model_megned_changed,myMLP=mod_mlp)
    checkpoint_path = 'best_models/double_train_AFLOW_on_%s_%s-v1.ckpt'%(model_to_test,nRuns)
    checkpoint = torch.load(checkpoint_path)
    lit_module_loaded = ModelLightningModule(model=new_model,loss=checkpoint['hyper_parameters']['loss'], lr=checkpoint['hyper_parameters']['lr'], scaler=checkpoint['hyper_parameters']['scaler'])
    lit_module_loaded.load_state_dict(checkpoint['state_dict'])
    model_best = lit_module_loaded.model
    with open('best_models/val_idx_%s_%s.pkl'%(model_to_test,nRuns), 'rb') as f:
        val_idx = pickle.load(f)
    
    return model_best,val_idx



dataset_name = 'MIX'    
model_to_test      = 'L96'

nRunsmax = 9

SetToUse, structure, scaler = return_dataset_test (dataset_name,model_to_test)


print("Test DataSet %s on %s double_train_AFLOW on %s"%(dataset_name,nRunsmax,model_to_test))
    
mapes_all = []
for nRuns in range (1,nRunsmax+1):

    new_model,val_idx = restore_model(model_to_test,nRuns)    
    new_model.train(False)
    mape_run =  mape_run_model (SetToUse, new_model, scaler, structure, val_idx, full_set=True)    
    mapes_all.append(mape_run)
    print("for model %s mape is %0.3f ;"%(nRuns,mape_run))

print("#between models MAPE: %0.2f (%0.2f)    #"%(np.array(mapes_all).mean(),np.array(mapes_all).std()))


