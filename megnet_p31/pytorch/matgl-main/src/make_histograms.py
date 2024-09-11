#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:23:11 2024

@author: lklochko
"""

import matplotlib.pyplot as plt
from cmcrameri import cm
import numpy as np
import seaborn as sns
import pandas as pd
# Data from the table
import re

# Data from the table
data = {
    'L96_1': [0.54, 2.53, 1.74, 2.16],
    'HH143_1': [0.51, 0.38, 0.43, 0.48],
    'MIX_1': [0.69, 0.77, 0.67, 0.97],
    'AFLOW_1': [0.55, 1.18, 0.93, 0.60],  # N/A values
    'L96_2': [0.51, 3.09, 2.07, 2.32],
    'HH143_2': [0.55, 0.37, 0.44, 0.52],
    'MIX_2': [0.66, 0.75, 0.71, 1.10],
    'AFLOW_2': [0.44, 1.27, 0.94, 0.49],
    'L96_3': [0.29, 1.26, 0.87, 0.57],
    'HH143_3': [0.55, 0.69, 0.63, 0.62],
    'MIX_3': [0.37, 0.78, 0.62, 0.59]
}

df = pd.DataFrame(data, index=['L96', 'HH143', 'MIX', 'AFLOW'])

# Create the heatmap
plt.figure(figsize=(12, 8), dpi=700)
ax = sns.heatmap(df.T, annot=True, cmap=cm.vik, fmt='.2f', linewidths=3,cbar_kws={'shrink': 1})
ax.collections[0].colorbar.set_label("MAPE",fontsize=20)
plt.ylabel('Train on', fontsize=20)
plt.xlabel('Train on', fontsize=20)

plt.xticks(rotation=45, fontsize=15)
plt.yticks(rotation=0, fontsize=15)
plt.show()

'''
cmap = cm.batlow  # Choose a color map, e.g., 'coolwarm'


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
plt.show()
'''