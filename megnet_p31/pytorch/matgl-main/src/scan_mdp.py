#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:02:54 2024

@author: lklochko
"""

# we need to save all structures of mdp in pkl file # 

from __future__ import annotations
import pandas as pd 
import torch 
import warnings 

warnings.simplefilter("ignore")



    

def unlog10(data):
        data_unlog10 = 10**(data)-1
        return data_unlog10
    
def inverse_transform(scalerY,X):
        if scalerY.mean is None or scalerY.std is None:
            raise ValueError("The StandardScaler has not been fitted yet.")
        return X * scalerY.std + scalerY.mean    



model_for_scan_scan  = 'HH143'
scalerY = torch.load('/home/lklochko/Desktop/ProjPostDoc/GitHub/fine_tuning_p60/megnet_p31/pytorch/matgl-main/src/torch.scaler.%s'%(model_for_scan_scan))

df = pd.read_pickle('mpd_ids_srtcuture_table.pkl')



nRunsmax = 9
idx = 1 

    



res = {}

for nRuns in range (1,nRunsmax+1):
    y_pred = [] 
    print("###=> le modele numero %s; "%(nRuns))
    model = torch.load('best_models/model_%s.%s'%(model_for_scan_scan,nRuns))
    model.train(False)     
    
    for idx in range(0,len(df)):
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

