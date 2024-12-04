import pandas as pd
from pymatgen.ext.matproj import MPRester
keyAPI = "add_your_key"
# =============================================================================
# For Laurent  
# =============================================================================
mpd_ids = pd.read_csv('id_compounds')

results = {'mpd_id': [], 'structure': []}
for ii in range(0,len(mpd_ids)):
    with MPRester(keyAPI) as mpr:
        docs = mpr.summary.search(material_ids=mpd_ids.names[ii],fields=["structure"])
        results['mpd_id'].append(mpd_ids.names[ii])
        results['structure'].append(docs[0].structure)
    
df = pd.DataFrame(results)
df.to_pickle('mpd_ids_srtcuture_table_newL.pkl')