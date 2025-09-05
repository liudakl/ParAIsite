# "results_scan/scan_for_double_train_AFLOW_on*.results.csv"
#%%

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import os

file_path = 'variances_paraisite.csv'

if os.path.exists(file_path):
    print(f"The file '{file_path}' exists.")
    df_final_updated = pd.read_csv('/home/lklochko/Desktop/ProjPostDoc/GitHub/ParAIsite/to_update_paper/variances_paraisite.csv')

else:
    nRunsmax = 9 

    # =============================================================================   
    # =============================================================================  
    model_for_scan_scan = 'Dataset1'


    step_1 = 'sample-'
    st1_model_1 = pd.DataFrame()


    df_new = pd.read_csv('results_scan/scan_for_%s__%s.results.csv'%(step_1,model_for_scan_scan))

    cs=[]
    for nRuns in range (1,nRunsmax+1):
        cs.append("res_"+str(nRuns))
        tds = df_new[cs].T
        meand =  tds.mean() 
        stdd =  tds.std()
        stdpd = stdd/meand
        maxd = tds.max()
        
        varianced = stdd**2
        cvd = stdd/meand

        df_new["mean"] = meand
        df_new["std"]  = stdd
        df_new["stdp"] = stdpd
        df_new["max"]  = maxd
        df_new["max.std"] = maxd*stdpd
        df_new["variance"] = varianced
        df_new["cv"] = cvd

    st1_model_1 = pd.concat([st1_model_1,df_new.drop(columns=['res_1', 'res_2', 'res_3', 'res_4', 'res_5', 'res_6',
        'res_7', 'res_8', 'res_9'])],axis=1).drop(columns='index')
    st1_model_1['step'] = 'step_1'


    step_2 = 'no_weights-'
    st2_model_1 = pd.DataFrame()


    df_new = pd.read_csv('results_scan/scan_for_%s__%s.results.csv'%(step_2,model_for_scan_scan))

    cs=[]
    for nRuns in range (1,nRunsmax+1):
        cs.append("res_"+str(nRuns))
        tds = df_new[cs].T
        meand =  tds.mean() 
        stdd =  tds.std()
        stdpd = stdd/meand
        maxd = tds.max()
        
        varianced = stdd**2
        cvd = stdd/meand

        df_new["mean"] = meand
        df_new["std"]  = stdd
        df_new["stdp"] = stdpd
        df_new["max"]  = maxd
        df_new["max.std"] = maxd*stdpd
        df_new["variance"] = varianced
        df_new["cv"] = cvd

    st2_model_1 = pd.concat([st2_model_1,df_new.drop(columns=['res_1', 'res_2', 'res_3', 'res_4', 'res_5', 'res_6',
        'res_7', 'res_8', 'res_9'])],axis=1).drop(columns='index')
    st2_model_1['step'] = 'step_2'


    step_3 = 'double_train_AFLOW_on_'
    st3_model_1 = pd.DataFrame()


    df_new = pd.read_csv('results_scan/scan_for_%s__%s.results.csv'%(step_3,model_for_scan_scan))

    cs=[]
    for nRuns in range (1,nRunsmax+1):
        cs.append("res_"+str(nRuns))
        tds = df_new[cs].T
        meand =  tds.mean() 
        stdd =  tds.std()
        stdpd = stdd/meand
        maxd = tds.max()
        
        varianced = stdd**2
        cvd = stdd/meand

        df_new["mean"] = meand
        df_new["std"]  = stdd
        df_new["stdp"] = stdpd
        df_new["max"]  = maxd
        df_new["max.std"] = maxd*stdpd
        df_new["variance"] = varianced
        df_new["cv"] = cvd

    st3_model_1 = pd.concat([st3_model_1,df_new.drop(columns=['res_1', 'res_2', 'res_3', 'res_4', 'res_5', 'res_6',
        'res_7', 'res_8', 'res_9'])],axis=1).drop(columns='index')
    st3_model_1['step'] = 'step_3'


    merged_df = pd.concat([st1_model_1, st2_model_1, st3_model_1], ignore_index=True)
    merged_df = merged_df.pivot(index='mpd_id', columns='step')
    merged_df.columns = [f'{col[0]}_{col[1]}' for col in merged_df.columns]
    merged_df_Dataset1 = merged_df.reset_index()


    # =============================================================================   
    # =============================================================================  
    model_for_scan_scan = 'Dataset2'


    step_1 = 'sample-'
    st1_model_1 = pd.DataFrame()


    df_new = pd.read_csv('results_scan/scan_for_%s__%s.results.csv'%(step_1,model_for_scan_scan))

    cs=[]
    for nRuns in range (1,nRunsmax+1):
        cs.append("res_"+str(nRuns))
        tds = df_new[cs].T
        meand =  tds.mean() 
        stdd =  tds.std()
        stdpd = stdd/meand
        maxd = tds.max()
        
        varianced = stdd**2
        cvd = stdd/meand

        df_new["mean"] = meand
        df_new["std"]  = stdd
        df_new["stdp"] = stdpd
        df_new["max"]  = maxd
        df_new["max.std"] = maxd*stdpd
        df_new["variance"] = varianced
        df_new["cv"] = cvd

    st1_model_1 = pd.concat([st1_model_1,df_new.drop(columns=['res_1', 'res_2', 'res_3', 'res_4', 'res_5', 'res_6',
        'res_7', 'res_8', 'res_9'])],axis=1).drop(columns='index')
    st1_model_1['step'] = 'step_1'


    step_2 = 'no_weights-'
    st2_model_1 = pd.DataFrame()


    df_new = pd.read_csv('results_scan/scan_for_%s__%s.results.csv'%(step_2,model_for_scan_scan))

    cs=[]
    for nRuns in range (1,nRunsmax+1):
        cs.append("res_"+str(nRuns))
        tds = df_new[cs].T
        meand =  tds.mean() 
        stdd =  tds.std()
        stdpd = stdd/meand
        maxd = tds.max()
        
        varianced = stdd**2
        cvd = stdd/meand

        df_new["mean"] = meand
        df_new["std"]  = stdd
        df_new["stdp"] = stdpd
        df_new["max"]  = maxd
        df_new["max.std"] = maxd*stdpd
        df_new["variance"] = varianced
        df_new["cv"] = cvd

    st2_model_1 = pd.concat([st2_model_1,df_new.drop(columns=['res_1', 'res_2', 'res_3', 'res_4', 'res_5', 'res_6',
        'res_7', 'res_8', 'res_9'])],axis=1).drop(columns='index')
    st2_model_1['step'] = 'step_2'


    step_3 = 'double_train_AFLOW_on_'
    st3_model_1 = pd.DataFrame()


    df_new = pd.read_csv('results_scan/scan_for_%s__%s.results.csv'%(step_3,model_for_scan_scan))

    cs=[]
    for nRuns in range (1,nRunsmax+1):
        cs.append("res_"+str(nRuns))
        tds = df_new[cs].T
        meand =  tds.mean() 
        stdd =  tds.std()
        stdpd = stdd/meand
        maxd = tds.max()
        
        varianced = stdd**2
        cvd = stdd/meand

        df_new["mean"] = meand
        df_new["std"]  = stdd
        df_new["stdp"] = stdpd
        df_new["max"]  = maxd
        df_new["max.std"] = maxd*stdpd
        df_new["variance"] = varianced
        df_new["cv"] = cvd

    st3_model_1 = pd.concat([st3_model_1,df_new.drop(columns=['res_1', 'res_2', 'res_3', 'res_4', 'res_5', 'res_6',
        'res_7', 'res_8', 'res_9'])],axis=1).drop(columns='index')
    st3_model_1['step'] = 'step_3'


    merged_df = pd.concat([st1_model_1, st2_model_1, st3_model_1], ignore_index=True)
    merged_df = merged_df.pivot(index='mpd_id', columns='step')
    merged_df.columns = [f'{col[0]}_{col[1]}' for col in merged_df.columns]
    merged_df_Dataset2 = merged_df.reset_index()


    # =============================================================================   
    # =============================================================================  
    model_for_scan_scan = 'MIX'


    step_1 = 'sample-'
    st1_model_1 = pd.DataFrame()


    df_new = pd.read_csv('results_scan/scan_for_%s__%s.results.csv'%(step_1,model_for_scan_scan))

    cs=[]
    for nRuns in range (1,nRunsmax+1):
        cs.append("res_"+str(nRuns))
        tds = df_new[cs].T
        meand =  tds.mean() 
        stdd =  tds.std()
        stdpd = stdd/meand
        maxd = tds.max()
        
        varianced = stdd**2
        cvd = stdd/meand

        df_new["mean"] = meand
        df_new["std"]  = stdd
        df_new["stdp"] = stdpd
        df_new["max"]  = maxd
        df_new["max.std"] = maxd*stdpd
        df_new["variance"] = varianced
        df_new["cv"] = cvd

    st1_model_1 = pd.concat([st1_model_1,df_new.drop(columns=['res_1', 'res_2', 'res_3', 'res_4', 'res_5', 'res_6',
        'res_7', 'res_8', 'res_9'])],axis=1).drop(columns='index')
    st1_model_1['step'] = 'step_1'


    step_2 = 'no_weights-'
    st2_model_1 = pd.DataFrame()


    df_new = pd.read_csv('results_scan/scan_for_%s__%s.results.csv'%(step_2,model_for_scan_scan))

    cs=[]
    for nRuns in range (1,nRunsmax+1):
        cs.append("res_"+str(nRuns))
        tds = df_new[cs].T
        meand =  tds.mean() 
        stdd =  tds.std()
        stdpd = stdd/meand
        maxd = tds.max()
        
        varianced = stdd**2
        cvd = stdd/meand

        df_new["mean"] = meand
        df_new["std"]  = stdd
        df_new["stdp"] = stdpd
        df_new["max"]  = maxd
        df_new["max.std"] = maxd*stdpd
        df_new["variance"] = varianced
        df_new["cv"] = cvd


    st2_model_1 = pd.concat([st2_model_1,df_new.drop(columns=['res_1', 'res_2', 'res_3', 'res_4', 'res_5', 'res_6',
        'res_7', 'res_8', 'res_9'])],axis=1).drop(columns='index')
    st2_model_1['step'] = 'step_2'


    step_3 = 'double_train_AFLOW_on_'
    st3_model_1 = pd.DataFrame()


    df_new = pd.read_csv('results_scan/scan_for_%s__%s.results.csv'%(step_3,model_for_scan_scan))

    cs=[]
    for nRuns in range (1,nRunsmax+1):
        cs.append("res_"+str(nRuns))
        tds = df_new[cs].T
        meand =  tds.mean() 
        stdd =  tds.std()
        stdpd = stdd/meand
        maxd = tds.max()
        
        varianced = stdd**2
        cvd = stdd/meand

        df_new["mean"] = meand
        df_new["std"]  = stdd
        df_new["stdp"] = stdpd
        df_new["max"]  = maxd
        df_new["max.std"] = maxd*stdpd
        df_new["variance"] = varianced
        df_new["cv"] = cvd

    st3_model_1 = pd.concat([st3_model_1,df_new.drop(columns=['res_1', 'res_2', 'res_3', 'res_4', 'res_5', 'res_6',
        'res_7', 'res_8', 'res_9'])],axis=1).drop(columns='index')
    st3_model_1['step'] = 'step_3'


    merged_df = pd.concat([st1_model_1, st2_model_1, st3_model_1], ignore_index=True)
    merged_df = merged_df.pivot(index='mpd_id', columns='step')
    merged_df.columns = [f'{col[0]}_{col[1]}' for col in merged_df.columns]
    merged_df_MIX = merged_df.reset_index()


    # =============================================================================   
    # =============================================================================  

    model_for_scan_scan = 'AFLOW'


    step_1 = 'sample-'
    st1_model_1 = pd.DataFrame()


    df_new = pd.read_csv('results_scan/scan_for_%s__%s.results.csv'%(step_1,model_for_scan_scan))

    cs=[]
    for nRuns in range (1,nRunsmax+1):
        cs.append("res_"+str(nRuns))
        tds = df_new[cs].T
        meand =  tds.mean() 
        stdd =  tds.std()
        stdpd = stdd/meand
        maxd = tds.max()
        
        varianced = stdd**2
        cvd = stdd/meand

        df_new["mean"] = meand
        df_new["std"]  = stdd
        df_new["stdp"] = stdpd
        df_new["max"]  = maxd
        df_new["max.std"] = maxd*stdpd
        df_new["variance"] = varianced
        df_new["cv"] = cvd

    st1_model_1 = pd.concat([st1_model_1,df_new.drop(columns=['res_1', 'res_2', 'res_3', 'res_4', 'res_5', 'res_6',
        'res_7', 'res_8', 'res_9'])],axis=1).drop(columns='index')
    st1_model_1['step'] = 'step_1'


    step_2 = 'no_weights-'
    st2_model_1 = pd.DataFrame()


    df_new = pd.read_csv('results_scan/scan_for_%s__%s.results.csv'%(step_2,model_for_scan_scan))

    cs=[]
    for nRuns in range (1,nRunsmax+1):
        cs.append("res_"+str(nRuns))
        tds = df_new[cs].T
        meand =  tds.mean() 
        stdd =  tds.std()
        stdpd = stdd/meand
        maxd = tds.max()
        
        varianced = stdd**2
        cvd = stdd/meand

        df_new["mean"] = meand
        df_new["std"]  = stdd
        df_new["stdp"] = stdpd
        df_new["max"]  = maxd
        df_new["max.std"] = maxd*stdpd
        df_new["variance"] = varianced
        df_new["cv"] = cvd

    st2_model_1 = pd.concat([st2_model_1,df_new.drop(columns=['res_1', 'res_2', 'res_3', 'res_4', 'res_5', 'res_6',
        'res_7', 'res_8', 'res_9'])],axis=1).drop(columns='index')
    st2_model_1['step'] = 'step_2'


    merged_df = pd.concat([st1_model_1, st2_model_1], ignore_index=True)
    merged_df = merged_df.pivot(index='mpd_id', columns='step')
    merged_df.columns = [f'{col[0]}_{col[1]}' for col in merged_df.columns]
    merged_df_AFLOW = merged_df.reset_index()


    # =============================================================================   
    # =============================================================================  

    merged_df_Dataset1['model'] = 'Dataset1'
    merged_df_Dataset2['model'] = 'Dataset2'
    merged_df_MIX['model'] = 'MIX'
    merged_df_AFLOW['model'] = 'AFLOW'

    final_merged_df = pd.concat([merged_df_Dataset1, merged_df_Dataset2, merged_df_MIX, merged_df_AFLOW], ignore_index=True)

    cols = final_merged_df.columns.tolist()
    cols = ['mpd_id', 'model'] + [col for col in cols if col not in ['mpd_id', 'model']]
    final_merged_df = final_merged_df[cols]


    final_merged_df = final_merged_df.fillna(0)

    df_melted = final_merged_df.melt(
        id_vars=['mpd_id', 'model'],
        var_name='step',
        value_name='value'
    )

    df_melted[['statistic', 'step']] = df_melted['step'].str.rsplit('_', n=1, expand=True)
    df_final = df_melted.pivot_table(
        index=['mpd_id', 'model', 'step'],
        columns='statistic',
        values='value'
    ).reset_index()
    df_final.columns.name = None

    mask = (df_final.model == 'AFLOW') & (df_final.step == '3')
    df_final_updated = df_final[~mask].copy()

    df_final_updated.to_csv('variances_paraisite.csv')



# =============================================================================  
#                                       PLOT 
# =============================================================================  

df_final_updated = df_final_updated.replace([np.inf, -np.inf], np.nan)
df_final_updated = df_final_updated.dropna()


custom_palette = {
    'AFLOW': '#1f77b4',  # Blue
    'MIX': '#ff7f0e',  # Orange
    'Dataset1': '#2ca02c',  # Green
    'Dataset2': "#a02c90",  # Green
}


# Calculate basic variance per database 

loaded_data = pd.read_pickle('structures_scalers/%s.pkl'%('AFLOW'))
df = loaded_data.dropna().copy()
var_AFLOW_0 = df.TC.var()

loaded_data = pd.read_pickle('structures_scalers/%s.pkl'%('MIX'))
df = loaded_data.dropna().copy()
var_MIX_0 = df.TC.var()


loaded_data = pd.read_pickle('structures_scalers/%s.pkl'%('Dataset1'))
df = loaded_data.dropna().copy()
var_Dataset1_0 = df.TC.var()

loaded_data = pd.read_pickle('structures_scalers/%s.pkl'%('Dataset2'))
df = loaded_data.dropna().copy()
var_Dataset2_0 = df.TC.var()

original_variances = {
    'AFLOW': var_AFLOW_0,
    'MIX': var_MIX_0,
    'Dataset1': var_Dataset1_0,
    'Dataset2': var_Dataset2_0,
}
fig, ax = plt.subplots(figsize=(10, 6))

sns.boxplot(data=df_final_updated[df_final_updated.model.isin(['Dataset1', 'AFLOW'])], x='step', y='variance_step', hue='model', palette=custom_palette,ax=ax)

plt.yscale('log')
plt.grid(True)
plt.title('Variance by Step and Model: TC > 100')
plt.ylabel(r'$Var$(mTC)')

plt.tight_layout()
plt.show()



# ==========


sns.scatterplot(
    data=df_final_updated[df_final_updated.model.isin(['Dataset1', 'AFLOW'])],
    x='mean_step',
    y='variance_step',
    hue='model',
    style='step',
    palette=custom_palette,
    s=100,
    alpha=0.7,
    markers={1: "o", 2: "^", 3: "P"}  # mapping step -> marker
)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Mean Predicted Thermal Conductivity (mTC)')
plt.ylabel(r'$Var$(mTC)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig("/home/lklochko/Desktop/ProjPostDoc/GitHub/ParAIsite/to_update_paper/paper_vria_allTC.pdf", format="pdf", dpi=100, bbox_inches="tight")
plt.show()


# =============================================================================  
#                   Correlation with materials properties
# ============================================================================= 

# path to save plots : /home/lklochko/Desktop/ProjPostDoc/GitHub/ParAIsite/to_update_paper

import seaborn as sns
from scipy.stats import pearsonr, spearmanr, f_oneway, ttest_ind
from cmcrameri import cm

metrics = ["cv_step", "max.std_step", "max_step", 
           "mean_step", "std_step", "stdp_step", "variance_step"]

df_wide = df_final_updated.pivot_table(
    index=["mpd_id", "model"], 
    columns="step", 
    values=metrics
)

df_wide.columns = [f"{metric}_{int(step)}" for metric, step in df_wide.columns]
df_wide = df_wide.reset_index()
df_wide = df_wide[df_wide.model=='Dataset1']

mpd_descriptors = pd.read_csv('all_mpd_decr.csv').dropna(axis=1).drop(columns='deprecated')
data = df_wide[['mpd_id','mean_step_1', 'mean_step_2', 'mean_step_3', 'variance_step_1', 'variance_step_2', 'variance_step_3']].sort_values(by='variance_step_1', ascending=True).head(101)

df = pd.merge(mpd_descriptors,data,on='mpd_id')

# deprecated is nan ... 

boolenas = df.select_dtypes(include='bool').columns
df[boolenas] = df[boolenas].astype(int)

numeric_cols = df.select_dtypes(include=np.number).columns
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

colum_to_correlate = 'mean_step_1'
numeric_features.remove(colum_to_correlate)

corr_results = []

for feature in numeric_features:
    pearson_corr, pearson_p = pearsonr(df[feature], df[colum_to_correlate])
    spearman_corr, spearman_p = spearmanr(df[feature], df[colum_to_correlate])
    corr_results.append({
        'feature': feature,
        'pearson_corr': pearson_corr,
        'pearson_p': pearson_p,
        'spearman_corr': spearman_corr,
        'spearman_p': spearman_p
    })

# for pearson corr : +1/-1 perfect line relation, 0 nothing 
# for pearson p : p<0.05 is statistically significant corr, other like a noise 


corr_df = pd.DataFrame(corr_results).dropna()
corr_df = corr_df.sort_values(by='pearson_corr', key=abs, ascending=False)
print(corr_df)
corr_df.to_csv('correlation_results_mean_TC.csv')


## correlation matrix for Dataset1
df_corr = df[numeric_cols].corr()
threshold = 0.2 
target_columns = ['mean_step_1', 'mean_step_2', 'mean_step_3']  
significant_features = df_corr.index[df_corr[target_columns].abs().max(axis=1) > threshold]
df_corr_significant = df_corr.loc[significant_features, significant_features]

plt.figure(figsize=(16,10))
sns.heatmap(df_corr, fmt=".2f", annot=True, cmap=cm.vikO)
plt.show()

