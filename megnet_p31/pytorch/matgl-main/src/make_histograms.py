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


data = {
    ('L96_nw'): [0.46, 2.24, 1.53, 2.45],
    ('HH143_nw'): [0.53, 0.37, 0.44, 0.50],
    ('MIX_nw'): [0.70, 0.73, 0.72, 1.03],
    ('AFLOW_I'): [0.49, 1.28, 0.96, 0.51],
    ('L96_w'): [0.52, 1.83, 1.30, 2.44],
    ('HH143_w'): [0.51, 0.37, 0.43, 0.50],
    ('MIX_w'): [0.63, 0.76, 0.71, 1.16],
    ('AFLOW_w'): [0.42, 1.17, 0.88, 0.51],
    ('L96_dt'): [0.29, 1.21, 0.85, 0.61],
    ('HH143_dt'): [0.58, 0.65, 0.63, 0.63],
    ('MIX_dt'): [0.34, 0.71, 0.56, 0.72]
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


