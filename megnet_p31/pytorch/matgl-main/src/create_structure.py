#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:38:18 2024

@author: lklochko
"""

import pandas as pd 
import numpy as np
from pymatgen.ext.matproj import MPRester


#df1 = pd.read_csv("https://gitlab.univ-lorraine.fr/klochko1/mdp/-/raw/main/cif_small_L.csv",index_col=0)
#df1.rename({'chemsys': 'formula','k_voigt':'kV', 'k_vrh':'kVRH', 'k_reuss':'kR','g_reuss':'gR','g_vrh':'gVRH','g_voigt':'gV'}, axis=1,inplace=True)

df2 = pd.read_csv("https://gitlab.univ-lorraine.fr/klochko1/mdp/-/raw/main/hh_143.csv", delimiter=';')
df2 = df2.reset_index(drop=True)

#df_1 = df1[['mpd_id','TC']]
df_2 = df2[['mpd_id','TC']]

#SetToUse = pd.concat([df_1,df_2], ignore_index=True)
SetToUse = df_2.copy()
SetToUse['mpd_id'] = SetToUse['mpd_id'].astype(str)
SetToUse['mpd_id'] = SetToUse['mpd_id'].apply(lambda x: x if x.startswith('mp-') else f'mp-{x}')


keyAPI = "0qnsciDAnjfIC8yrYYpz5bUmjgAZHH2p" 
mpr = MPRester(keyAPI)


structure = []
for i in SetToUse.mpd_id:
    structure.append(mpr.get_structure_by_material_id(i))


import pickle
with open('structures_HH143.pkl', 'wb') as f:
    pickle.dump(structure, f)