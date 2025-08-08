#%%
from __future__ import annotations
import pandas as pd 
import warnings
import torch 
import lightning as pl
from custom_functions import unlog10, inverse_transform, model_to_scan
import json 
import sys 
import tqdm


warnings.simplefilter("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"
torchseed = 42 
pl.seed_everything(torchseed, workers=True)
torch.manual_seed(torchseed)
torch.cuda.manual_seed(torchseed)



with open('for_scan.json') as f:
   params = json.load(f)


nRunsmax            = params['Number_of_RUNS']
NN1                 = params['Layer1_NN']
NN2                 = params['Layer2_NN']
NN3                 = params['Layer3_NN']
NN4                 = params['Layer4_NN']


step = sys.argv[1]
model_for_scan_scan = sys.argv[2]


data_to_test        = 'mpd_ids_srtcuture_table'



# =============================================================================
#                               SETUP MODEL TO SCAN
#   
# =============================================================================  

scalerY = torch.load('structures_scalers/torch.scaler.%s'%(model_for_scan_scan))
loaded_data = pd.read_pickle('/home/lklochko/Desktop/ProjPostDoc/GitHub/ParAIsite/nopush/structures_scalers/%s.pkl'%(data_to_test))
df = loaded_data.dropna().copy()


if step == 'step1': step = 'sample-'
if step == 'step2': step = 'no_weights-'
if step == 'step3': step = 'double_train_AFLOW_on_'

res = {}

for nRuns in range (1,nRunsmax+1):
    print('We do Run:', nRuns)
    y_pred = [] 
    checkpoint_path = 'best_models/%s%s_%s.ckpt'%(step,model_for_scan_scan,nRuns)
    model =  model_to_scan(checkpoint_path,nRuns,device,NN1,NN2,NN3,NN4)
    model.train(False) 
        
    for idx in range(0,len(df)):
        if df['structure'].iloc[idx] != [] :
            print(idx)
            preds = model.predict_structure(df['structure'].iloc[idx])
            preds_ivT =  inverse_transform(scalerY,preds)
            tc_pred = unlog10(preds_ivT).item()
            y_pred.append(tc_pred)
    res["res_"+str(nRuns)] = y_pred


resdf = pd.DataFrame(res)
df_new = pd.concat([df.reset_index(), resdf], axis=1).set_index("index").dropna().drop(columns='structure')
df_new.to_csv("results_scan/scan_for_%s__%s.results.csv"%(step,model_for_scan_scan))

