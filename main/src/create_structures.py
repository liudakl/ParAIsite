import pandas as pd
from pymatgen.ext.matproj import MPRester
import sys 

if len(sys.argv) != 2:
   print("please provide a config file")
   sys.exit(-1)

data_set_name = sys.argv[1] 

# =============================================================================
#   EXAMPLE OF CREATING DATA FOR SCAN
# =============================================================================


keyAPI = "add_your_key"

mpd_ids = pd.read_csv(data_set_name)
results = {'mpd_id': [], 'structure': []}

for ii in range(0,len(mpd_ids)):
    with MPRester(keyAPI) as mpr:
        docs = mpr.summary.search(material_ids=mpd_ids.names[ii],fields=["structure"])
        results['mpd_id'].append(mpd_ids.names[ii])
        results['structure'].append(docs[0].structure)
    
df = pd.DataFrame(results)
df.to_pickle('%s_for_scan.pkl'%(data_set_name))

# =============================================================================
#   HOW TO PREPARE YOUT OWN DATA ? 
# =============================================================================

