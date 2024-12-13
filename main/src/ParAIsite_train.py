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
from custom_functions import return_dataset_train,create_changed_megned_model, setup_dataset
import matgl 

warnings.simplefilter("ignore")

# =============================================================================
#                               SETUP DATASET TO TRAIN
#   
# =============================================================================    


dataset_name_TRAIN = 'Dataset1'

if dataset_name_TRAIN != 'MIX':
    _, mp_dataset = setup_dataset (dataset_name_TRAIN)   
else: 
    _, mp_dataset,mp_dataset_1, mp_dataset_2  = setup_dataset (dataset_name_TRAIN) 

scaler = torch.load('structures_scalers/torch.scaler')

# =============================================================================
#                               SETUP MODEL 
#   
# =============================================================================

if torch.cuda.is_available():
     device = 'cuda'
     accelerator = 'gpu'
else: 
     device = 'cpu'
     accelerator = 'cpu'

best_mapes = [] 
maxRuns = 1
maxEpochs = 300
NN1 = 450
NN2 = 350
NN3 = 350
NN4 = 0
torchseed = 42 
learning_rate = 1e-3
test_data = 0

pl.seed_everything(torchseed, workers=True)
torch.manual_seed(torchseed)
torch.cuda.manual_seed(torchseed)
    

for nRuns in range (1,maxRuns+1):
    best_mape = np.inf
    checkpoint_callback = ModelCheckpoint(monitor='val_Total_Loss',dirpath='best_models/',filename='sample-%s_%s'%(dataset_name_TRAIN,nRuns))
    
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
    
    megnet_loaded = matgl.load_model("MEGNet-MP-2018.6.1-Eform").to(device)
    model_megned_changed =  create_changed_megned_model() 
    model_megned_changed.load_state_dict(megnet_loaded.state_dict(),strict=False)
    mod_mlp = myMLP (16,NN1,NN2,NN3,NN4,1).to(device)
    new_model = combined_models(pretrained_model=model_megned_changed,myMLP=mod_mlp).to(device)
    lit_module = ModelLightningModule(model=new_model,loss='l1_loss',lr=learning_rate,scaler=scaler).to(device)

# =============================================================================
#               TRAINING OF THE MODEL 
# =============================================================================

    logger = CSVLogger("logs", name="MEGNet_m1_training_%s_%s"%(dataset_name_TRAIN,nRuns),version=0)
    trainer = pl.Trainer(max_epochs=maxEpochs,devices="auto", accelerator=accelerator, logger=logger,callbacks=[checkpoint_callback])
    trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)



# =============================================================================
#               TEST OF THE RESULTS
# =============================================================================


    if bool(test_data): 
        
        # =============================================================================
        #                               SETUP DATASET TO TEST
        #   
        # =============================================================================

        dataset_name_test1 = 'Dataset2'
        dataset_name_test2 = 'Dataset1'
        dataset_name_test4 = 'AFLOW'



        res_tes1_Dataset2,mp_dataset_test1 = setup_dataset(dataset_name_test1) 
        res_tes2_Dataset1,mp_dataset_test2 = setup_dataset(dataset_name_test2) 
        res_tes4_AFLOW, mp_dataset_test4 = setup_dataset(dataset_name_test4) 
        res_tes3_MIX = []

        try:   
            os.remove("structures_scalers/torch.scaler")
        except FileNotFoundError:
            pass
    
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





# =============================================================================
#                           METRICS CHECK UP  
# =============================================================================


    metrics = pd.read_csv("logs/MEGNet_m1_training_%s_%s/version_0/metrics.csv"%(dataset_name_TRAIN,nRuns))

    x1 = metrics["train_Total_Loss"].dropna().reset_index().drop(columns='index')
    x2 = metrics["val_Total_Loss"].dropna().reset_index().drop(columns='index')   
        
    min_mape_val = x2.val_Total_Loss.min()
    
    if min_mape_val < best_mape:
        best_mape = min_mape_val
        best_mapes.append(best_mape)      
      
    for fn in ("dgl_graph.bin", "lattice.pt", "dgl_line_graph.bin", "state_attr.pt", "labels.json"):
        try:
            os.remove(fn)
        except FileNotFoundError:
            pass

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


if bool(test_data):
    df_final  = pd.DataFrame({
    'Run': range(1, maxRuns + 1),
    'train_on': '%s'%(dataset_name_TRAIN),
    'test_Dataset2': res_tes1_Dataset2,
    'test_Dataset1': res_tes2_Dataset1,
    'test_MIX': res_tes3_MIX,
    'test_AFLOW': res_tes4_AFLOW
})

    df_final.to_csv('results_on_train_test/results_with_weights_trained_on_%s.csv'%(dataset_name_TRAIN), index=False)




