#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 11:25:20 2024

@author: lklochko
"""

import requests
from pymatgen.core import  Structure



# ==== Fetch AFLOW ===== # 

path_to_save = '/home/lklochko/Desktop/ProjPostDoc/GitHub/fine_tuning_p60/megnet_p31/pytorch/matgl-main/src/aflow_cif'

response = requests.get("https://aflow.org/API/aflux/?$catalog(ICSD),agl_thermal_conductivity_300K(*),$paging(1,10000)")
json_res = response.json()

structure = [] 
TC = [] 

for idx in range(0,len(json_res)): 
    aurl = json_res[idx]['aurl'].replace('.edu:', '.edu/')
    entry = aurl.split('/')[-1] 
    file_name = f"{entry}.cif"
    download_url = f"http://{aurl}/{file_name}"
    response = requests.get(download_url)
    if response.status_code == 200:
        with open('%s/%s'%(path_to_save,file_name), 'wb') as file:
            file.write(response.content)
            print(f"File {file_name} downloaded successfully.")
    else:
            print(f"Failed to download the file. Status code: {response.status_code}")
    
    structure.append(Structure.from_file('%s/%s'%(path_to_save,file_name)))
    TC.append(json_res[idx]['agl_thermal_conductivity_300K'])




import pickle
with open('structures_AFLOW.pkl', 'wb') as f:
    pickle.dump(structure, f)
    
with open('TC_AFLOW.pkl', 'wb') as f:
    pickle.dump(TC, f)    