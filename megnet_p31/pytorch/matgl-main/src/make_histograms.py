#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:23:11 2024

@author: lklochko
"""

import pandas as pd
import matplotlib.pyplot as plt
from cmcrameri import cm


# Data for training
train_data = {
    'Dataset': ['L96', 'HH143', 'MIX'],
    'Validation Error (Mean)': [0.47, 0.34, 0.50],
    'Validation Error (Std)': [0.20, 0.07, 0.11]
}

# Data for testing L96
test_l96 = {
    'Tested on': ['HH143', 'MIX', 'L96 (validation set)', 'L96 (full set)'],
    'Error (Mean)': [1.45, 1.01, 0.47, 0.34],
    'Error (Std)': [0.47, 0.24, 0.20, 0.19]
}

# Data for testing HH143
test_hh143 = {
    'Tested on': ['L96', 'MIX', 'HH143 (validation set)', 'HH143 (full set)'],
    'Error (Mean)': [0.97, 0.52, 0.34, 0.22],
    'Error (Std)': [0.08, 0.06, 0.07, 0.10]
}

# Data for testing MIX
test_mix = {
    'Tested on': ['L96', 'HH143', 'MIX (validation set)', 'MIX (full set)'],
    'Error (Mean)': [0.47, 0.27, 0.50, 0.35],
    'Error (Std)': [0.21, 0.11, 0.11, 0.15]
}


train_df = pd.DataFrame(train_data)
test_l96_df = pd.DataFrame(test_l96)
test_hh143_df = pd.DataFrame(test_hh143)
test_mix_df = pd.DataFrame(test_mix)

train_df['Validation Error (Mean)'].plot(kind='hist', bins=5, alpha=0.7)
plt.title('Histogram of Validation Error (Mean) - Training')
plt.xlabel('Validation Error (Mean)')
plt.ylabel('Frequency')
plt.show()

# Plot histograms for test results on L96
test_l96_df['Error (Mean)'].plot(kind='hist', bins=5, alpha=0.7)
plt.title('Histogram of Error (Mean) - Test on L96')
plt.xlabel('Error (Mean)')
plt.ylabel('Frequency')
plt.show()

# Plot histograms for test results on HH143
test_hh143_df['Error (Mean)'].plot(kind='hist', bins=5, alpha=0.7)
plt.title('Histogram of Error (Mean) - Test on HH143')
plt.xlabel('Error (Mean)')
plt.ylabel('Frequency')
plt.show()

# Plot histograms for test results on MIX
test_mix_df['Error (Mean)'].plot(kind='hist', bins=5, alpha=0.7)
plt.title('Histogram of Error (Mean) - Test on MIX')
plt.xlabel('Error (Mean)')
plt.ylabel('Frequency')
plt.show()
