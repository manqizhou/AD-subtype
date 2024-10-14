import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
from pandas.tseries.offsets import DateOffset
import sys
import os
import copy
import ast

temp_data_path = sys.argv[1]#'./data_preparation/Temp_data_exclude_dementia/'
KM_result_path = sys.argv[2]
mapping_path = sys.argv[3]#'./data_preparation/mapping/'
raw_diag_file_path = sys.argv[4]
raw_diag_varNames = ast.literal_eval(sys.argv[5])
save_path = sys.argv[6]#'./cluster_characteristics/'

#print(raw_diag_varNames)
diag_id_varName=raw_diag_varNames[0]#PATID
diag_date_varName=raw_diag_varNames[1]#ADMIT_DATE
diag_type_varName=raw_diag_varNames[2]#DX_TYPE
diag_type9_name=raw_diag_varNames[3]# a list of values indiciting it's ICD9 code
diag_type10_name=raw_diag_varNames[4]#a list
diag_value_varName=raw_diag_varNames[5]#DX

###################### Part 2: count comorbidity before MCI ##############
# ### key comorbidities
comorbidities=['Hypertension','Hyperlipidemia','Diabetes','Dementias','Memory loss','Heart disease',
               'Sleep disorders','Anxiety','GERD','Cerebrovascular disease',
               'Chronic airway obstruction','Chronic renal failure','Urinary tract infection','Glaucoma and Cataract']

# comorbidities=['anemias',"Parkinson's disease","obesity",'schizophrenia','Vitamin D deficiency','pneumonia',
#     'malignant neoplasm','Vitamin B12 deficiency','Bipolar','decubitus','atherosclerosis',
#     'fatigue','hypothyroidism','tobacco use disorder','seizures']

####### first part: get comorbidity codes from name
diag_code2index = np.load(temp_data_path+'diag_code2index.npy', allow_pickle=True).item()
diag_code2name = np.load(temp_data_path+'diag_code2name.npy', allow_pickle=True).item()
save_path_comor=save_path+'comorbidity/'
if not os.path.exists(save_path_comor):
    os.makedirs(save_path_comor)
#######
val_list=list(diag_code2name.values())
key_list = list(diag_code2name.keys())

comorbidities_dict={}

# ### get code for key comorbidities
out=open(save_path_comor+'comorbidity_name_code.txt','w')
for comr in comorbidities:
    out.write(comr+'\n')
    if comr == 'Glaucoma and Cataract':
        m1 = [s for s in val_list if 'Glaucoma'.lower() in s.lower()]
        m2 = [s for s in val_list if 'Cataract'.lower() in s.lower()]
        matching = m1+m2
    else:
        matching = [s for s in val_list if comr.lower() in s.lower()]
    code=[]
    for i in matching:
        pos = val_list.index(i)
        c=key_list[pos]
        code.append(c)
    comorbidities_dict[comr]=code

out.close()
# comorbidities_dict['Hypertension'].remove('402')#remove 'Elevated blood pressure reading without diagnosis of hypertension'
with open(save_path_comor+'comorbidity_code.pickle','wb') as f1:
    pickle.dump(comorbidities_dict, f1)
f1.close()

#### second part: count number
df_diagnosis = pd.read_csv(raw_diag_file_path, dtype=str) # the raw full data
df_time = pd.read_csv(temp_data_path+'Patient_ID_MCI_before_AD.csv', index_col=0)

if 'APATID' in df_time.columns:
    patid_varName='APATID'
elif 'PATID' in df_time.columns:
    patid_varName = 'PATID'
else:
    print('Error in the Patient_ID_MCI_before_AD.csv column name: no PATID or APATID')
    sys.exit()
#
if 'BMCI_TIME' in df_time.columns:
    mci_time_varName='BMCI_TIME'
elif 'MCI_TIME' in df_time.columns:
    mci_time_varName = 'MCI_TIME'
else:
    print('Error in the Patient_ID_MCI_before_AD.csv column name: no MCI_TIME or BMCI_TIME')
    sys.exit()

#
df_icd9_to_phecode = pd.read_csv(mapping_path+'phecode_icd9_rolled.csv', dtype=str)
df_icd10_to_phecode = pd.read_csv(mapping_path+'phecode_icd10.csv', dtype=str)
dic_icd9_to_phecode = dict(zip(list(df_icd9_to_phecode.ICD9), list(df_icd9_to_phecode.PheCode)))
dic_icd10_to_phecode = dict(zip(list(df_icd10_to_phecode.ICD10), list(df_icd10_to_phecode.PheCode)))
dic_icd10_to_phecode['G31.84']='292.2'

comorbidities_dict=pickle.load(open(save_path_comor+'comorbidity_code.pickle','rb'))

###
comorbidities_names=list(comorbidities_dict.keys())
comorbidities_before_MCI={k: 0 for k in comorbidities_names}

# zhenxing add: start...., build a dic for saving the comorbidities of each cluster
p_id_cluster_data = pd.read_csv(KM_result_path+'MCI2AD_time_cluster_df.csv',index_col=False)
p_id_cluster_list = list(p_id_cluster_data['APATID'])
num_clusters=np.max(p_id_cluster_data.Cluster)

comorbidities_before_MCI_cluster ={}
#num_clusters = 5
for i in range(num_clusters):
    comorbidities_before_MCI_cluster['cluster_'+str(i+1)] = comorbidities_before_MCI.copy()
# zhenxing add: end.

for patient in tqdm(p_id_cluster_list): # zhenxing revise
    p_cluster_label = p_id_cluster_data.loc[p_id_cluster_data.APATID == patient,'Cluster'].values[0] # zhenxing add

    diag_one_patient = df_diagnosis.loc[df_diagnosis[diag_id_varName] == patient]
    time_MCI = pd.to_datetime(np.array(df_time.loc[df_time[patid_varName] == patient][mci_time_varName]))
    start_time = time_MCI - DateOffset(months=6)
    diag_one_patient_before_MCI = diag_one_patient.loc[(pd.to_datetime(diag_one_patient[diag_date_varName])>=start_time[0]) & (pd.to_datetime(diag_one_patient[diag_date_varName])<time_MCI[0])]
    #get ICD9/10 code
    icd9 = list(diag_one_patient_before_MCI.loc[diag_one_patient_before_MCI[diag_type_varName].isin(diag_type9_name)][diag_value_varName])
    icd10 = list(diag_one_patient_before_MCI.loc[diag_one_patient_before_MCI[diag_type_varName].isin(diag_type10_name)][diag_value_varName])

    #convert to phecode
    phecode=[]
    for code in icd9:
        if code in dic_icd9_to_phecode:
            phecode.append(dic_icd9_to_phecode[code])
        elif code[0:-1] in dic_icd9_to_phecode:
            phecode.append(dic_icd9_to_phecode[code[0:-1]])
        else:
            codetmp = code.split('.')[0]
            if codetmp in dic_icd9_to_phecode:
                phecode.append(dic_icd9_to_phecode[codetmp])
            else:
                pass
    for code in icd10:
        if code in dic_icd10_to_phecode:
            phecode.append(dic_icd10_to_phecode[code])
        else:
            codetmp = code.split('.')[0]
            if codetmp in dic_icd10_to_phecode:
                phecode.append(dic_icd10_to_phecode[codetmp])
            else:
                codetmp = code.split('.')[0]
                if codetmp in dic_icd10_to_phecode:
                    phecode.append(dic_icd10_to_phecode[codetmp])
                elif codetmp+'.0' in dic_icd10_to_phecode:
                    phecode.append(dic_icd10_to_phecode[codetmp+'.0'])
                else:
                    pass
    #find if ICD9/10 are in the patient's record
    for c in comorbidities_names:
        if len(set(comorbidities_dict[c]).intersection(phecode)) > 0:#
            # comorbidities_before_MCI[c]+=1
            comorbidities_before_MCI_cluster['cluster_'+str(p_cluster_label)][c] += 1 # zhenxing add
        else:
            pass

with open(save_path_comor+'comorbidity_count.pickle','wb') as f:
    pickle.dump(comorbidities_before_MCI_cluster, f) # comorbidities_before_MCI
f.close()


out=open(save_path_comor+'comorbidity_stats.txt','w')
for i in range(num_clusters):
    out.write('cluster_' + str(i + 1) + '\n')
    cluster_size = len(p_id_cluster_data.loc[p_id_cluster_data.Cluster == (i+1)])
    dic_cluster = comorbidities_before_MCI_cluster['cluster_'+str(i+1)]
    new_dic_cluster = {key: str((value / cluster_size)*100)[:5] for key, value in dic_cluster.items()}
    for m in comorbidities_names:
        out.write('%s\t%i\t%s\n' % (m,dic_cluster[m],new_dic_cluster[m]))

out.close()

