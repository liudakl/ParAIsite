#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:02:54 2024

@author: lklochko
"""

# we need to save all structures of mdp in pkl file # 

from __future__ import annotations
import pandas as pd 
import warnings
import torch 
import lightning as pl
from matgl.models import combined_models
from matgl.utils.training import ModelLightningModule
import matgl 
from model_mlp import myMLP

from custom_functions import create_changed_megned_model 

warnings.simplefilter("ignore")


def unlog10(data):
        data_unlog10 = 10**(data)-1
        return data_unlog10
    
def inverse_transform(scalerY,X):
        if scalerY.mean is None or scalerY.std is None:
            raise ValueError("The StandardScaler has not been fitted yet.")
        return X * scalerY.std + scalerY.mean    



model_for_scan_scan  = 'HH143'
scalerY = torch.load('structures_scalers/torch.scaler.%s'%(model_for_scan_scan))

df = pd.read_pickle('structures_scalers/mpd_ids_srtcuture_table.pkl')

torchseed = 42 
pl.seed_everything(torchseed, workers=True)
torch.manual_seed(torchseed)
torch.cuda.manual_seed(torchseed)

nRunsmax = 9

def model_to_scan (model_scan,nRuns):
    NN1 = 450
    NN2 = 350
    NN3 = 350
    NN4 = 0
    megnet_loaded = matgl.load_model("MEGNet-MP-2018.6.1-Eform").cuda()
    model_megned_changed =  create_changed_megned_model() .cuda()
    model_megned_changed.load_state_dict(megnet_loaded.state_dict(),strict=False)
    mod_mlp = myMLP (16,NN1,NN2,NN3,NN4,1).cuda()
    new_model = combined_models(pretrained_model=model_megned_changed,myMLP=mod_mlp).cuda()
    
    
    checkpoint_path = 'best_models/double_train_AFLOW_on_%s_%s.ckpt'%(model_scan,nRuns)
    lit_module_loaded = ModelLightningModule.load_from_checkpoint(checkpoint_path,model=new_model).cuda()
    return lit_module_loaded.model

res = {}

for nRuns in range (1,nRunsmax+1):
    y_pred = [] 
    print("###=> le modele numero %s; "%(nRuns))
    model =  model_to_scan (model_for_scan_scan,nRuns).cuda()
    model.train(False)     
    
    for idx in range(0,len(df)):
        if len (df['structure'].iloc[idx]) <= 3: 
            preds = model.predict_structure(df['structure'].iloc[idx])
            preds_ivT =  inverse_transform(scalerY,preds)
            tc_pred = unlog10(preds_ivT).item()
            y_pred.append(tc_pred)
        res["res_"+str(nRuns)] = y_pred


resdf = pd.DataFrame(res)
df_new = pd.concat([df.reset_index(), resdf], axis=1).set_index("index").dropna()

cs=[]
for nRuns in range (1,nRunsmax+1):
    cs.append("res_"+str(nRuns))
    tds = df_new[cs].T
    meand =  tds.mean() 
    stdd =  tds.std()
    stdpd = stdd/meand
    maxd = tds.max()

    df_new["mean"] = meand
    df_new["std"]  = stdd
    df_new["stdp"] = stdpd
    df_new["max"]  = maxd
    df_new["max.std"] = maxd*stdpd

print(df_new.loc[df_new["max"]<1.0, ["mpd_id","mean","std","stdp","max","max.std"]].sort_values("stdp"))
df_new.to_csv("stable_scan.MDP.model_%s.results.csv"%(model_for_scan_scan))

