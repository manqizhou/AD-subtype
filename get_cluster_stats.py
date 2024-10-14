######## This code is to get age/sex/race/PT for the each cluster
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import os
import copy

temp_data_path = sys.argv[1]#'./data_preparation/Temp_data_exclude_dementia/'
KM_result_path = sys.argv[2]
save_path = sys.argv[3]#'./cluster_characteristics/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

######################### Part 1: get age, sex, gender and PT ##############
###  load data
df_demo = pd.read_csv(temp_data_path + 'demo_condition_minLength_2_Age_Larger_than_50.csv')  # contain age/sex/race/PT
p_id_cluster_data = pd.read_csv(KM_result_path+'MCI2AD_time_cluster_df.csv', index_col=False)  # contain cluster label and patient_id
p_id_cluster_list = list(p_id_cluster_data['APATID'])
num_clusters=np.max(p_id_cluster_data.Cluster)
# save
out = open(save_path + 'cluster_stats.txt', 'w')
stats_cluster_dict = {}# create dict to save results
stats_dict = {'age': [], 'race': [], 'gender': [], 'PT': []}
for i in range(num_clusters):
    stats_cluster_dict['cluster_' + str(i + 1)] = copy.deepcopy(stats_dict)

#### decide variable name
if 'HMCI2AD_time' in df_demo.columns:
    mci2ad_time_varName='HMCI2AD_time'
elif 'MCI2AD_time' in df_demo.columns:
    mci2ad_time_varName = 'MCI2AD_time'
else:
    print('Error in the demo column name: no MCI2AD_time or HMCI2AD_time')
    sys.exit()
#
if 'DSEX' in df_demo.columns:
    sex_varName='DSEX'
elif 'SEX' in df_demo.columns:
    sex_varName = 'SEX'
else:
    print('Error in the demo column name: no DSEX or SEX')
    sys.exit()
#
if 'GMCIAGE' in df_demo.columns:
    age_varName='GMCIAGE'
elif 'MCIAGE' in df_demo.columns:
    age_varName = 'MCIAGE'
else:
    print('Error in the demo column name: no GMCIAGE or MCIAGE')
    sys.exit()
#
if 'FRACE' in df_demo.columns:
    race_varName='FRACE'
elif 'RACE' in df_demo.columns:
    race_varName = 'RACE'
else:
    print('Error in the demo column name: no FRACE or RACE')
    sys.exit()
#
if 'APATID' in df_demo.columns:
    patid_varName='APATID'
elif 'PATID' in df_demo.columns:
    patid_varName = 'PATID'
else:
    print('Error in the demo column name: no PATID or APATID')
    sys.exit()

##########
for patient in tqdm(p_id_cluster_list):  # zhenxing revise
    p_cluster_label = p_id_cluster_data.loc[p_id_cluster_data.APATID == patient, 'Cluster'].values[0]  # zhenxing add
    info_one_patient = df_demo.loc[df_demo[patid_varName] == patient]

    stats_cluster_dict['cluster_' + str(p_cluster_label)]['age'].append(info_one_patient[age_varName].item())
    stats_cluster_dict['cluster_' + str(p_cluster_label)]['race'].append(info_one_patient[race_varName].item())
    stats_cluster_dict['cluster_' + str(p_cluster_label)]['gender'].append(info_one_patient[sex_varName].item())

    pt = info_one_patient[mci2ad_time_varName].item()
    if (pt >= 365) and (pt < 365 * 2):
        pt_name = '1~2'
    elif (pt >= 365 * 2) and (pt < 365 * 3):
        pt_name = '2~3'
    elif (pt >= 365 * 3) and (pt < 365 * 4):
        pt_name = '3~4'
    elif (pt >= 365 * 4) and (pt < 365 * 5):
        pt_name = '4~5'
    else:
        pt_name = '>5'

    stats_cluster_dict['cluster_' + str(p_cluster_label)]['PT'].append(pt_name)

for i in range(num_clusters):
    out.write('cluster_' + str(i + 1)+'\n')
    dict_sub = stats_cluster_dict['cluster_' + str(i + 1)]
    # number
    Total_num = len(dict_sub['age'])
    out.write('Total number: %i\n' % Total_num)
    # age
    mci_age = dict_sub['age']
    mci_age_median = np.median(np.array(mci_age))
    mci_age_mean = np.mean(np.array(mci_age))
    out.write('Age: %s [%s,%s]\n' % (
    str(mci_age_median)[0:5], str(np.percentile(mci_age, 25)), str(np.percentile(mci_age, 75))))
    # sex
    female = (np.array(dict_sub['gender']) == 'F').sum()
    out.write('Female: %i (%s)\n' % (female, str(female / Total_num*100)[0:5]))
    # race
    race = np.array(dict_sub['race'])
    count = np.unique(race, return_counts=True)
    for i in range(count[0].shape[0]):
        out.write('Race %s: %i (%s)\n' % (count[0][i], count[1][i], str(count[1][i] / Total_num*100)[0:5]))
    # PT
    PT = np.array(dict_sub['PT'])
    count = np.unique(PT, return_counts=True)
    for i in range(count[0].shape[0]):
        out.write('PT %s: %i (%s)\n' % (count[0][i], count[1][i], str(count[1][i] / Total_num*100)[0:5]))

out.close()

