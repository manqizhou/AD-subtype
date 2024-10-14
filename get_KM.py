############################## This code is used to get supp fig 14 plot data ##############################
###
########################################################################################################################
import numpy as np
import pandas as pd
#import scipy.io as sio
#from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
#from matplotlib.pyplot import MultipleLocator
import sys
import os

######## load data
cluster_path=sys.argv[1]
temp_data_path=sys.argv[2]
save_path=sys.argv[3]#

if not os.path.exists(save_path):
    os.makedirs(save_path)

####
Data = np.load(f'{cluster_path}T_GBTM_result_all.npy', allow_pickle=True)
Probability = Data[2]

demo = pd.read_csv(f'{temp_data_path}demo_condition_minLength_2_Age_Larger_than_50.csv')
Patient_label = np.load(f'{temp_data_path}Patient_label_diag.npy')
Time_sheet = np.load(f'{temp_data_path}time_sheet_condition_minLength_2_Age_Larger_than_50.npy')
Time_sheet = Time_sheet/12

###
MCI2AD_time = []
p_id_list = []

## decide MCI2AD_time variable name
if 'HMCI2AD_time' in demo.columns:
    mci2ad_time_varName='HMCI2AD_time'
elif 'MCI2AD_time' in demo.columns:
    mci2ad_time_varName = 'MCI2AD_time'
else:
    print('Error in the demo column name: no MCI2AD_time or HMCI2AD_time')
    sys.exit()

## decide PATID variable name
if 'APATID' in demo.columns:
    patid_varName='APATID'
elif 'PATID' in demo.columns:
    patid_varName = 'PATID'
else:
    print('Error in the demo column name: no PATID or APATID')
    sys.exit()

for i in range(demo.shape[0]):
    index = (Patient_label == i)
    if (demo.iloc[i, :][mci2ad_time_varName] >= 365 * 1) and ((Time_sheet[index] <= 5).sum() > 2):
        MCI2AD_time.append(demo.iloc[i,:][mci2ad_time_varName])
        p_id_list.append(demo.iloc[i,:][patid_varName])

#######################################################################################################################
num_cluster = Probability.shape[1]

# Zhenxing add: generate df for plot survival figure using Kaplan Meier
MCI2AD_time_cluster_df = pd.DataFrame(columns=['APATID', 'MCI2AD_time', 'Cluster', 'AD'])
#MCI2AD_time_cluster_df = pd.DataFrame(columns=['MCI2AD_time', 'Cluster', 'AD'])

p_cluster = []
p_id_cluster_s = []
p_MCI2AD_time = []
p_AD_label = []

for c in range(num_cluster):
    MCI2AD_time_cluster = []
    p_id_cluster = []
    for i in range(Probability.shape[0]):
        if Probability[i].argmax() == c:
            MCI2AD_time_cluster.append(MCI2AD_time[i])
            p_id_cluster.append(p_id_list[i])

    # zhx add: start
    #print('cluster',c)
    #print('len(MCI2AD_time_cluster)',len(MCI2AD_time_cluster))
    p_id_cluster_s = p_id_cluster_s + p_id_cluster
    p_cluster = p_cluster + [str(c+1)]*len(MCI2AD_time_cluster)
    p_MCI2AD_time = p_MCI2AD_time + MCI2AD_time_cluster
    p_AD_label = p_AD_label + [1]*len(MCI2AD_time_cluster) # 1 indicate AD label, all patients are AD
    # zhx add: end

# Zhenxing: generate df for plot survival figure using Kaplan Meier
MCI2AD_time_cluster_df['APATID'] = p_id_cluster_s
MCI2AD_time_cluster_df['MCI2AD_time'] = p_MCI2AD_time
MCI2AD_time_cluster_df['Cluster'] = p_cluster
MCI2AD_time_cluster_df['AD'] = p_AD_label
MCI2AD_time_cluster_df.to_csv(save_path+'MCI2AD_time_cluster_df.csv',index=False)
#print('DONE.')
