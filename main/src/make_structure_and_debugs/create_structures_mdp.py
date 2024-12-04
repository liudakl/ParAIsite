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


with MPRester(keyAPI) as mpr:
    docs = mpr.summary.search(is_stable=True,fields=["material_id", "structure"])

mpids = [doc.material_id for doc in docs]
structure = [doc.structure for doc in docs]


df = pd.DataFrame({
    'mpd_id': mpids,
    'structure': structure
})

df.to_pickle('mpd_ids_srtcuture_table_newL.pkl')

import pickle
with open('mpd_structure_stable.pkl', 'wb') as f:
    pickle.dump(structure, f)
    
    

# =============================================================================
# For Laurent  
# =============================================================================
mpd_ids = pd.read_csv('make_structure_and_debugs/id_compounds')

results = {'mpd_id': [], 'structure': []}
for ii in range(0,len(mpd_ids)):
    with MPRester(keyAPI) as mpr:
        docs = mpr.summary.search(material_ids=mpd_ids.names[ii],fields=["structure"])
        results['mpd_id'].append(mpd_ids.names[ii])
        results['structure'].append(docs[0].structure)
    
df = pd.DataFrame(results)
df.to_pickle('mpd_ids_srtcuture_table_newL.pkl')
        
#mpids = [doc.material_id for doc in docs]
#structure = [doc.structure for doc in docs]        
    