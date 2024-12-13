from __future__ import annotations
import pandas as pd 
import warnings
import torch 
import lightning as pl
from custom_functions import unlog10, inverse_transform, model_to_scan
import json 
import sys 

warnings.simplefilter("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"
torchseed = 42 
pl.seed_everything(torchseed, workers=True)
torch.manual_seed(torchseed)
torch.cuda.manual_seed(torchseed)


with open(sys.argv[1]) as f:
   params = json.load(f)

nRunsmax            = params['Number_of_RUNS']
NN1                 = params['Layer1_NN']
NN2                 = params['Layer2_NN']
NN3                 = params['Layer3_NN']
NN4                 = params['Layer4_NN']
model_for_scan_scan = params['model_for_scan_scan']
data_to_test        = params['data_to_test']


# =============================================================================
#                               SETUP MODEL TO SCAN
#   
# =============================================================================  

scalerY = torch.load('structures_scalers/torch.scaler.%s'%(model_for_scan_scan))
loaded_data = pd.read_pickle('structures_scalers/%s.pkl'%(data_to_test))
df = loaded_data.dropna().copy()



res = {}

for nRuns in range (1,nRunsmax+1):
    y_pred = [] 
    checkpoint_path = 'best_models/double_train_AFLOW_on_%s_%s.ckpt'%(model_for_scan_scan,nRuns)
    model =  model_to_scan(checkpoint_path,nRuns,device,NN1,NN2,NN3,NN4)
    model.train(False) 
        
    for idx in range(0,len(df)):
        if df['structure'].iloc[idx] != [] :
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
df_new.to_csv("results_scan/scan_for.data_%s.results.csv"%(data_to_test))
