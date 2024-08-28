#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:56:39 2024

@author: lklochko
"""
from __future__ import annotations
import warnings
from matgl.utils.training import ModelLightningModule
import matgl 
from model_mlp import myMLP
import numpy as np
import pickle 
from matgl.models import combined_models
from custom_functions import return_dataset_test,create_changed_megned_model,mape_run_model


warnings.simplefilter("ignore")


def restore_model (model_to_test,nRuns,double_traine,set_weights):
    NN1 = 450
    NN2 = 350
    NN3 = 350
    NN4 = 0
    #megnet_loaded = matgl.load_model("/Users/liudmylaklochko/Desktop/FineTuningRemote/fine_tuning_papers/megnet_p31/pytorch/matgl-main/pretrained_models/MEGNet-MP-2018.6.1-Eform")
    megnet_loaded = matgl.load_model("MEGNet-MP-2018.6.1-Eform")
    model_megned_changed =  create_changed_megned_model() 
    model_megned_changed.load_state_dict(megnet_loaded.state_dict(),strict=False)
    mod_mlp = myMLP (16,NN1,NN2,NN3,NN4,1)
    new_model_restore = combined_models(pretrained_model=model_megned_changed,myMLP=mod_mlp)
    if set_weights == True: 
         checkpoint_path = 'best_models/sample-%s_%s.ckpt'%(model_to_test,nRuns)
         with open('best_models/val_idx_%s_%s.pkl'%(model_to_test,nRuns), 'rb') as f:
             val_idx = pickle.load(f)
    elif double_traine == True:
         checkpoint_path = 'best_models/double_train_AFLOW_on_%s_%s-v3.ckpt'%(model_to_test,nRuns)
         with open('best_models/double_val_idx_AFLOW_on_%s_%s.pkl'%(model_to_test,nRuns), 'rb') as f:
             val_idx = pickle.load(f)
    else:
        checkpoint_path = 'best_models/no_weights-%s_%s.ckpt'%(model_to_test,nRuns)
        with open('best_models/val_idx_%s_%s.pkl'%(model_to_test,nRuns), 'rb') as f:
            val_idx = pickle.load(f)
    lit_module_loaded = ModelLightningModule.load_from_checkpoint(checkpoint_path,model=new_model_restore)
    
    model_best = lit_module_loaded.model
    
    
    
    
    
    return model_best,val_idx
    


double_traine = False
set_weights   = False
full_set      = True
 
model_to_test = 'HH143'
dataset_name  = 'AFLOW'    


nRunsmax = 9

SetToUse, structure, scaler = return_dataset_test (dataset_name,model_to_test)

if   set_weights   == True :
        print("Test DataSet %s on %s pre-trained models on %s with MEGNET weights "%(dataset_name,nRunsmax,model_to_test))
elif double_traine == True:
        print("Test DataSet %s on %s pre-trained models on %s"%(dataset_name,nRunsmax,model_to_test))
else: 
        print("Test DataSet %s on %s pre-trained models on %s WITHOUT MEGNET weights "%(dataset_name,nRunsmax,model_to_test))
    
mapes_all = []
for nRuns in range (1,nRunsmax+1):

    new_model,val_idx = restore_model(model_to_test,nRuns,double_traine=double_traine,set_weights=set_weights)    
    new_model.train(False)
    mape_run =  mape_run_model (SetToUse, new_model, scaler, structure, val_idx, full_set=full_set)    
    mapes_all.append(mape_run)
    print("for model %s mape is %0.3f ;"%(nRuns,mape_run))

print("#between models MAPE: %0.2f (%0.2f)    #"%(np.array(mapes_all).mean(),np.array(mapes_all).std()))



