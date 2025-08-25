# "results_scan/scan_for_double_train_AFLOW_on*.results.csv"
#%%

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

custom_palette = {
    'AFLOW': '#1f77b4',  # Blue
    'MIX': '#ff7f0e',  # Orange
    'Dataset1': '#2ca02c',  # Green
    'Dataset2': "#a02c90",  # Green
}

fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df_final_updated[df_final_updated['mean_step']<3.0], x='step', y='variance_step', hue='model', palette=custom_palette)
plt.yscale('log')
plt.grid(True)
plt.title('Variance by Step and Model: TC < 3.0')
plt.show()


# ==========


plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=df_final_updated[df_final_updated['mean_step']<3.0],
    x='mean_step',
    y='variance_step',
    hue='model',
    style='step',
    palette=custom_palette,
    s=100,  # size of the points
    alpha=0.7
)
plt.xlabel('Predicted Thermal Conductivity (TC)')
plt.ylabel('Variance_step')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# ========== Collecting info from MPD ====== 


