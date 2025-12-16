from mp_api.client import MPRester
import pandas as pd 


mpdkey = ''

Material_Project_Stable = pd.read_pickle('/home/lklochko/Desktop/ProjPostDoc/GitHub/ParAIsite/nopush/structures_scalers/mpd_ids_srtcuture_table.pkl')
ids  = Material_Project_Stable.mpd_id

def extractElements(formula):
    """
    Extracts elements from a chemical formula string.
    """
    e = ""
    es = []
    for c in formula:
        if c < "a" and e != "":
            es.append(e)
            e = ""
        if not c.isdigit(): e += c
    if e: es.append(e)
    return es


def describeMaterial(mpid, mpdkey):
    """
    Returns the description of a material from its MP ID.
    Requires the MPDKey to access the Materials Project Database.
    """
    with MPRester(mpdkey, use_document_model=False) as mpr:
        # Get the material summary document from the Materials Project
        # using the provided material ID (mpid).
        # and returning all available fields.
        # If the document is not found, return None.
        if not mpid.startswith("mp-"):
            mpid = "mp-" + mpid
        doc = mpr.materials.summary.search(material_ids=[mpid], fields=mpr.materials.summary.available_fields)
        if not doc or len(doc) == 0: return None
        doc = doc[0]
        mat = doc["material_id"]
        formula = doc.get("formula_pretty", None)
        elements = []
        if formula: elements = extractElements(formula)
        description = {"material_id": mat, "formula": formula, "elements": elements, "descriptors": {}}
        for field in mpr.materials.summary.available_fields:
            if field in doc:
                value = doc[field]
                if isinstance(value, (float, bool, int)):
                    description["descriptors"][field] = value
                # else: print("Field {} has non-numeric value: {}".format(field, value))
        return description
    

all_data = []
k =  0 
for mpid in Material_Project_Stable['mpd_id']:
    mat = describeMaterial(str(mpid), mpdkey)
    if mat is None:
        print(f"Warning: mpid {mpid} returned None, skipping")
        continue  # skip to next ID

    row = {
        'material_id': mat.get('material_id'),
        'formula': mat.get('formula'),
        'elements': ','.join(mat.get('elements', []))  
    }
    row.update(mat.get('descriptors', {}))
    
    all_data.append(row)
    print(k)
    k += 1

df = pd.DataFrame(all_data)

df.to_csv('all_mpd_decr.csv')
