#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:02:54 2024

@author: lklochko
"""

# we need to save all structures of mdp in pkl file # 
import pandas as pd
from pymatgen.ext.matproj import MPRester
keyAPI = "0qnsciDAnjfIC8yrYYpz5bUmjgAZHH2p" 
mpr = MPRester(keyAPI)

'''
structure = [] 
with MPRester(keyAPI) as mpr:
    docs = mpr.summary.search(fields=["material_id", "structure"])

mpids = [doc.material_id for doc in docs]
structure = [doc.structure for doc in docs]
'''




import pickle
with open('mpd_structure.pkl', 'wb') as f:
    pickle.dump(structure, f)