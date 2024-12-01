#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:23:11 2024

@author: lklochko
"""

import matplotlib.pyplot as plt
from cmcrameri import cm
import seaborn as sns
import pandas as pd
# Data from the table

'''
# Data from the table
data = {
    'Dataset1_1': [0.54, 2.53, 1.74, 2.16],
    'Dataset2_1': [0.51, 0.38, 0.43, 0.48],
    'MIX_1': [0.69, 0.77, 0.67, 0.97],
    'AFLOW_1': [0.55, 1.18, 0.93, 0.60],  # N/A values
    'Dataset1_2': [0.51, 3.09, 2.07, 2.32],
    'Dataset2_2': [0.55, 0.37, 0.44, 0.52],
    'MIX_2': [0.66, 0.75, 0.71, 1.10],
    'AFLOW_2': [0.44, 1.27, 0.94, 0.49],
    'Dataset1_3': [0.29, 1.26, 0.87, 0.57],
    'Dataset2_3': [0.55, 0.69, 0.63, 0.62],
    'MIX_3': [0.37, 0.78, 0.62, 0.59]
}

df = pd.DataFrame(data, index=['Dataset1', 'Dataset2', 'MIX', 'AFLOW'])

data_L96 = {
    'L96_1': [0.54],
    'L96_2': [0.51],
    'L96_3': [0.29],

}

df_L96 = pd.DataFrame(data_L96, index=['L96'])



# Create the heatmap
plt.figure(figsize=(12, 10), dpi=700)
ax = sns.heatmap(df.T, annot=True, cmap=cm.vik, fmt='.2f', linewidths=3,cbar_kws={'shrink': 1})
ax.collections[0].colorbar.set_label("MAPE",fontsize=15)
plt.ylabel('Train on', fontsize=15)
plt.xlabel('Test on', fontsize=15)

plt.xticks(rotation=45, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.savefig('/home/lklochko/Desktop/ProjPostDoc/GitHub/ParAIsite/megnet_p31/pytorch/matgl-main/src/heatmap_results.eps', format='eps')
plt.show()


'''
#colors = ['skyblue', 'lightsalmon', 'lightpink', 'lightblue', 'lightyellow', 'lightgrey', 'cyan']


df = pd.read_csv('../../../../../paper/reg_results_TC.csv')
df[['model', 'target']] = df['Unnamed: 0'].str.split(' ', n=1, expand=True)
df[['mape_mean', 'mape_std']] = df['TC'].str.extract(r'([0-9.]+) \(([^)]+)\)')
df['mape_mean'] = pd.to_numeric(df['mape_mean'])
df['mape_std'] = pd.to_numeric(df['mape_std'])
df = df.drop(columns=['TC'])
df = df.drop(columns=['Unnamed: 0'])
df = df[['model', 'target', 'mape_mean','mape_std']]
df['model'] = df['model'].replace('paper31', 'MEGNet')
df['model'] = df['model'].replace('paper49', 'ALIGNN')
df['model'] = df['model'].replace('paper60', 'CrabNet')
cmap = cm.batlow  # Choose a color map, e.g., 'coolwarm'

mask = df['mape_mean'] < 0.51

df['Label'] = df['model'] + ' - ' + df['target']
df = df.sort_values(by='mape_mean')[mask]


norm = plt.Normalize(df['mape_mean'].max()-0.07, df['mape_mean'].max())
colors = [cmap(norm(value)) for value in df['mape_mean']]



plt.figure(figsize=(10, 8),dpi=700)
bars = plt.barh(df['Label'], df['mape_mean'], xerr=df['mape_std'], color=colors, ecolor='red', capsize=14)
for bar, target in zip(bars, df['target']):
    width = bar.get_width()
    plt.text(0.02, bar.get_y() + bar.get_height() /1.4,
             target, va='center', ha='left', color='black', fontsize=15)
plt.yticks([])

# Display Paper names on the left-hand side (axis)
plt.yticks(ticks=range(len(df)), labels=df['model'],fontsize=15)
plt.xticks(fontsize=15)

plt.xlabel('MAPE',fontsize=15)
plt.savefig('/home/lklochko/Desktop/ProjPostDoc/GitHub/ParAIsite/megnet_p31/pytorch/matgl-main/src/mape_mlp_test.png', format='png')

plt.show()





dataset_name_TRAIN = 'L96'
maxRuns = 9

y_temp_1  = 0 
y_temp_2  = 0 
y_temp_3  = 0 

m = 0 
for nRuns in range (1,maxRuns+1):
    df = pd.read_csv("logs/MEGNet_training_no_weights_%s_%s/version_0/metrics.csv"%(dataset_name_TRAIN,nRuns))
    y_temp_1 = df["val_Total_Loss"].dropna().reset_index().drop(columns='index')  + y_temp_1
    df = pd.read_csv("logs/MEGNet_m1_training_%s_%s/version_0/metrics.csv"%(dataset_name_TRAIN,nRuns))
    y_temp_2 = df["val_Total_Loss"].dropna().reset_index().drop(columns='index')  + y_temp_2
    df = pd.read_csv("logs/MEGNet_m1_best_model_double_training_%s_%s/version_0/metrics.csv"%(dataset_name_TRAIN,nRuns))
    y_temp_3 = df["val_Total_Loss"].dropna().reset_index().drop(columns='index')  + y_temp_3
    m = m + 1
    
y_v1 = y_temp_1 / m    
y_v2 = y_temp_2 / m 
y_v3 = y_temp_3 / m 

x = range(len(y_v1))
plt.figure(figsize=(14, 8),dpi=700)

#plt.plot(x, y_v1,label='step 1')
#plt.plot(x, y_v2,label='step 2')
#plt.plot(x, y_v3,label='step 3')

plt.scatter(x, y_v1,label='step 1',marker='o',s=80,color='tomato')
plt.scatter(x, y_v2,label='step 2',marker='^',s=80,color='darkcyan')
plt.scatter(x, y_v3,label='step 3',marker='+',s=300,color='olivedrab')

plt.ylabel('Validation Loss', fontsize=20)
plt.xlabel('Epochs', fontsize=20)


plt.xticks(rotation=45, fontsize=15)
plt.yticks(rotation=0, fontsize=15)

plt.legend(fontsize=15)
if dataset_name_TRAIN == 'L96':
    name_title = 'Dataset1'
elif dataset_name_TRAIN == 'HH143':
    name_title = 'Dataset2'
    
plt.title('%s database'%(name_title), fontsize=15)

plt.savefig('/home/lklochko/Desktop/ProjPostDoc/GitHub/ParAIsite/megnet_p31/pytorch/matgl-main/src/loss_all_steps_%s.eps'%(dataset_name_TRAIN), format='eps')

plt.show()


'''

dataset_name_TRAIN = 'L96'
maxRuns = 9

y_temp_1  = 0 
y_temp_2  = 0 
y_temp_3  = 0 

m = 0 
plt.figure(figsize=(14, 8),dpi=700)
for nRuns in range (1,4+1):
    df = pd.read_csv("logs/MEGNet_training_no_weights_%s_%s/version_0/metrics.csv"%(dataset_name_TRAIN,nRuns))
    y_temp_1 = df["val_Total_Loss"].dropna().reset_index().drop(columns='index')  
    x = range(len(y_temp_1))
    plt.scatter(x, y_temp_1,label='sample %s'%(nRuns),marker='o',s=5)#,color='tomato')
    min_loss = y_temp_1.min()
    min_loss_idx = y_temp_1.idxmin().values[0]
    plt.scatter(min_loss_idx, min_loss, color='red', s=100, facecolors='none', edgecolors='red', label='Min Run %s' % nRuns)
plt.ylabel('Validation Loss', fontsize=20)
plt.xlabel('Epochs', fontsize=20)


plt.xticks(rotation=45, fontsize=15)
plt.yticks(rotation=0, fontsize=15)

plt.legend(fontsize=15)
if dataset_name_TRAIN == 'L96':
    name_title = 'Dataset1'
elif dataset_name_TRAIN == 'HH143':
    name_title = 'Dataset2'

plt.title('%s database'%(name_title), fontsize=15)


'''




dataset_name_TRAIN = 'AFLOW'
maxRuns = 9

y_temp_1_list = []
y_temp_2_list = []
y_temp_3_list = []

for nRuns in range(1, maxRuns + 1):
    df = pd.read_csv(f"logs/MEGNet_training_no_weights_{dataset_name_TRAIN}_{nRuns}/version_0/metrics.csv")
    y_temp_1_list.append(df["val_Total_Loss"].dropna().reset_index(drop=True))

    df = pd.read_csv(f"logs/MEGNet_m1_training_{dataset_name_TRAIN}_{nRuns}/version_0/metrics.csv")
    y_temp_2_list.append(df["val_Total_Loss"].dropna().reset_index(drop=True))

#    df = pd.read_csv(f"logs/MEGNet_m1_best_model_double_training_{dataset_name_TRAIN}_{nRuns}/version_0/metrics.csv")
#    y_temp_3_list.append(df["val_Total_Loss"].dropna().reset_index(drop=True))

df_y1 = pd.concat(y_temp_1_list, axis=1)
df_y2 = pd.concat(y_temp_2_list, axis=1)
#df_y3 = pd.concat(y_temp_3_list, axis=1)

y_v1_mean = df_y1.mean(axis=1)
y_v2_mean = df_y2.mean(axis=1)
#y_v3_mean = df_y3.mean(axis=1)

y_v1_std = df_y1.std(axis=1)
y_v2_std = df_y2.std(axis=1)
#y_v3_std = df_y3.std(axis=1)


idx1 = y_v1_mean.idxmin()
idx2 = y_v2_mean.idxmin()
#idx3 = y_v3_mean.idxmin()


print('step I for model %s: %s(%s)'%(dataset_name_TRAIN,y_v1_mean.min(), y_v1_std.iloc[y_v1_mean.idxmin()]))
print('step II for model %s: %s(%s)'%(dataset_name_TRAIN,y_v2_mean.min(), y_v2_std.iloc[y_v2_mean.idxmin()]))

#print('step III for model %s: %s(%s)'%(dataset_name_TRAIN,y_v3_mean.min(), y_v3_std.iloc[y_v3_mean.idxmin()]))


