#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:23:11 2024

@author: lklochko
"""

import pandas as pd
import matplotlib.pyplot as plt
from cmcrameri import cm
import seaborn as sns

# Data for training

train_m0_dataset  = {
    'Dataset': ['L96', 'HH143', 'MIX','AFLOW'],
    'Validation Error (Mean)': [0.51,0.35,0.48,0.61]
}


train_m1_dataset = {
    'Dataset': ['L96', 'HH143', 'MIX','AFLOW'],
    'Validation Error (Mean)': [0.47, 0.34, 0.50,0.49]
}






train_m0_df = pd.DataFrame(train_m0_dataset)
#train_df = train_df.set_index(['Dataset'])

train_m1_df = pd.DataFrame(train_m1_dataset)

corr_matrix = train_m0_df.corrwith(other=train_m1_df, axis=1)

plt.figure(figsize=(10, 6), dpi=1250)
sns.heatmap(corr_matrix, annot=True,cmap=cm.roma)
plt.show()
#import matplotlib.pyplot as mpl
#import numpy as np
#m = np.array([[.6, .3, .2], [.8, .4, .9]])
#mpl.imshow(m, cmap=cm.managua)
#mpl.colorbar()
#mpl.show()







'''
hatches = ['//', '++', 'xx']
#colors = ['lightcoral', 'forestgreen', 'steelblue']
colors = ['lawngreen','lawngreen','lawngreen']
plt.figure(figsize=(10, 6), dpi=350)
bars = plt.bar(train_df['Dataset'], train_df['Validation Error (Mean)'],  color=colors, edgecolor='darkslategrey', alpha=0.7)

for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)
    bar.set_edgecolor('black') 



legend_labels = ['MEGNET']
handles = [plt.Line2D([0], [0], color=color, lw=4, label=label) for color, label in zip(colors, legend_labels)]
plt.legend(handles=handles, title='Pre-trained on', fontsize=12, title_fontsize=14, loc='center',bbox_to_anchor=(0.5,0.90))


colors = ['papayawhip','papayawhip','papayawhip']

bars = plt.bar(test_aflow_df['Dataset'], test_aflow_df['Validation Error (Mean)'],  color=colors, edgecolor='darkslategrey', alpha=0.7)

for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)
    bar.set_edgecolor('black') 



legend_labels = ['AFLOW']
handles = [plt.Line2D([0], [0], color=color, lw=4, label=label) for color, label in zip(colors, legend_labels)]
plt.legend(handles=handles, title='Pre-trained on', fontsize=12, title_fontsize=14, loc='center',bbox_to_anchor=(0.5,0.90))

plt.tight_layout(rect=[0, 0, 0.85, 1]) 
plt.xlabel('Model', fontsize=16)
plt.ylabel('Mean Average Percentage Error', fontsize=16)
plt.xticks(rotation=45, ha="right", fontsize=14) 
plt.yticks(fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout() 
plt.savefig('/home/lklochko/Desktop/ProjPostDoc/GitHub/fine_tuning_p60/megnet_p31/pytorch/matgl-main/src/high_quality_plot.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1, transparent=False)
plt.show()


'''





