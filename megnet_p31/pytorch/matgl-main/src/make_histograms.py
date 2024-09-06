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


# Data from the table
data = {
    'L96_nw': [0.54, 2.53, 1.74, 2.16],
    'HH143_nw': [0.51, 0.38, 0.43, 0.48],
    'MIX_nw': [0.69, 0.77, 0.67, 0.97],
    #'AFLOW_nw': [None, None, None, None],  # N/A values
    'L96_w': [0.51, 3.09, 2.07, 2.32],
    'HH143_w': [0.55, 0.37, 0.44, 0.52],
    'MIX_w': [0.66, 0.75, 0.71, 1.10],
    'AFLOW_w': [0.44, 1.27, 0.94, 0.49],
    'L96_dt': [0.29, 1.26, 0.87, 0.57],
    'HH143_dt': [0.55, 0.69, 0.63, 0.62],
    'MIX_dt': [0.37, 0.78, 0.62, 0.59]
}


# Convert the data to a DataFrame
df = pd.DataFrame(data, index=['L96', 'HH143', 'MIX', 'AFLOW'])

# Plotting the heatmap
plt.figure(figsize=(8, 6),dpi=700)
sns.heatmap(df.T, annot=True, cmap=cm.vik, fmt='.2f', linewidths=0.5)
plt.title('Validation Errors Heatmap (Train vs. Test)')
plt.xlabel('Test on')
plt.ylabel('Train on')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()


