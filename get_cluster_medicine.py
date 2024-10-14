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
drug_rxnorm_codes_f=sys.argv[4]#'../cohort_stats/medicine/drug_rxnorm_codes.pickle'

raw_medicine_file_path = sys.argv[5]
raw_medicine_varNames = ast.literal_eval(sys.argv[6])

save_path = sys.argv[7]#'./cluster_characteristics/'

med_id_varName=raw_medicine_varNames[0]#PATID
med_date_varName=raw_medicine_varNames[1]#ADMIT_DATE
med_type_varName=raw_medicine_varNames[2]#DX_TYPE
med_typeNDC_name=raw_medicine_varNames[3]# a list of values indiciting it's ICD9 code
med_typeRX_name=raw_medicine_varNames[4]#a list
med_value_varName=raw_medicine_varNames[5]#DX

##
num_med_f=len(eval(raw_medicine_file_path))#number of input medicine files
if num_med_f==2:# oneFlorida case
    raw_dispense_file_path=eval(raw_medicine_file_path)[0]
    raw_prescribe_file_path=eval(raw_medicine_file_path)[1]
    df_dispense = pd.read_csv(raw_dispense_file_path, dtype=str)
    df_prescribe = pd.read_csv(raw_prescribe_file_path, dtype=str)

elif num_med_f==1:# UCSF/INSIGHT case
    raw_medicine_file_path=eval(raw_medicine_file_path)[0]
    df_medicine = pd.read_csv(raw_medicine_file_path, dtype=str)
else:
    print('error in the raw_medicine_file_path')

##### first part: get medicine code
save_path_med=save_path+'medicine/'
if not os.path.exists(save_path_med):
    os.makedirs(save_path_med)

drug_rxnorm_codes=pickle.load(open(drug_rxnorm_codes_f,'rb'))
## remove prefix
for k in drug_rxnorm_codes.keys():
    one_list=drug_rxnorm_codes[k]
    one_list_new=[i.split('RxNorm:')[-1] for i in one_list]
    drug_rxnorm_codes[k]=one_list_new

drug_code2index = np.load(temp_data_path+'drug_code2index.npy', allow_pickle=True).item()
all_drug_codes=list(drug_code2index.keys())

##
medicines_dict={}
for k in drug_rxnorm_codes.keys():
    all_ref_codes=list(drug_rxnorm_codes[k])
    code_list=[]
    for i in all_ref_codes:
        if i in all_drug_codes:
            code_list.append(i)
    medicines_dict[k]=code_list

with open(save_path_med+'medicine_code.pickle','wb') as f1:
    pickle.dump(medicines_dict, f1)
f1.close()

###### second part: count the number for medicine
#demo=pd.read_csv(temp_data_path+'demo_condition_minLength_2_Age_Larger_than_50.csv',dtype=str)
df_time = pd.read_csv(temp_data_path+'Patient_ID_MCI_before_AD.csv', index_col=0)

if 'APATID' in df_time.columns:
    patid_varName='APATID'
elif 'PATID' in df_time.columns:
    patid_varName = 'PATID'
else:
    print('Error in the Patient_ID_MCI_before_AD.csv column name: no PATID or APATID')
    sys.exit()

if 'BMCI_TIME' in df_time.columns:
    mci_time_varName='BMCI_TIME'
elif 'MCI_TIME' in df_time.columns:
    mci_time_varName = 'MCI_TIME'
else:
    print('Error in the Patient_ID_MCI_before_AD.csv column name: no MCI_TIME or BMCI_TIME')
    sys.exit()


## mapping file
df_ndc_to_rxcui = pd.read_csv(mapping_path+'NDC_RXNorm_mapping.csv', dtype=str)
dic_ndc_to_rxcui = dict(zip(list(df_ndc_to_rxcui.NDC), list(df_ndc_to_rxcui.rxnrom)))

##
medicines_dict=pickle.load(open(save_path_med+'medicine_code.pickle','rb'))

medicine_names=list(medicines_dict.keys())
medicine_before_MCI={k: 0 for k in medicine_names}#save count reulst

p_id_cluster_data = pd.read_csv(KM_result_path+'MCI2AD_time_cluster_df.csv',index_col=False)
p_id_cluster_list = list(p_id_cluster_data['APATID'])
num_clusters=np.max(p_id_cluster_data.Cluster)

medicine_before_MCI_cluster ={}

for i in range(num_clusters):
    medicine_before_MCI_cluster['cluster_'+str(i+1)] = copy.deepcopy(medicine_before_MCI)

for patient in tqdm(p_id_cluster_list):
    p_cluster_label = p_id_cluster_data.loc[p_id_cluster_data.APATID == patient, 'Cluster'].values[0]  # zhenxing add

    if num_med_f==2:#oneFlorida case
        dispense_one_patient = df_dispense.loc[df_dispense.PATID == patient]
        prescribe_one_patient = df_prescribe.loc[df_prescribe.PATID == patient]
        time_MCI = pd.to_datetime(np.array(df_time.loc[df_time[patid_varName] == patient][mci_time_varName]))
        start_time = time_MCI - DateOffset(months=6)
        dispense_one_patient_before_MCI = dispense_one_patient.loc[
            (pd.to_datetime(dispense_one_patient.DISPENSE_DATE) >= start_time[0]) &
            (pd.to_datetime(dispense_one_patient.DISPENSE_DATE) < time_MCI[0])]
        prescribe_one_patient_before_MCI = prescribe_one_patient.loc[
            (pd.to_datetime(prescribe_one_patient.RX_ORDER_DATE) >= start_time[0]) &
            (pd.to_datetime(prescribe_one_patient.RX_ORDER_DATE) < time_MCI[0])]
        # get code
        code_ndc = list(dispense_one_patient_before_MCI.NDC)
        code_rxnorm = list(prescribe_one_patient_before_MCI.RXNORM_CUI)
    else:## INSIGHT/UCSF case, only one medicine file
        medicine_one_patient = df_medicine.loc[df_medicine[med_id_varName] == patient]
        time_MCI = pd.to_datetime(np.array(df_time.loc[df_time[patid_varName] == patient][mci_time_varName]))
        start_time = time_MCI - DateOffset(months=6)
        medicine_one_patient_before_MCI = medicine_one_patient.loc[
            (pd.to_datetime(medicine_one_patient[med_date_varName]) >= start_time[0]) &
            (pd.to_datetime(medicine_one_patient[med_date_varName]) < time_MCI[0])]
        # get code
        code_ndc = list(
            medicine_one_patient_before_MCI.loc[medicine_one_patient_before_MCI[med_type_varName].isin(med_typeNDC_name)][
                med_value_varName])
        code_rxnorm = list(
            medicine_one_patient_before_MCI.loc[medicine_one_patient_before_MCI[med_type_varName].isin(med_typeRX_name)][
                med_value_varName])

    # conver NDC to RXnorm code
    code_ndc2rxnorm = []
    for code in code_ndc:
        if code in dic_ndc_to_rxcui:
            if pd.isnull(dic_ndc_to_rxcui[code]):  # if is na
                pass
            else:
                code_ndc2rxnorm.append(dic_ndc_to_rxcui[code])
        else:
            pass
    one_patient_drug_code = code_rxnorm + code_ndc2rxnorm

    for m in medicine_names:
        if len(set(medicines_dict[m]).intersection(one_patient_drug_code)) > 0:
            medicine_before_MCI_cluster['cluster_' + str(p_cluster_label)][m] += 1
        else:
            pass

# save count
with open(save_path_med+'medicine_count.pickle','wb') as f:
    pickle.dump(medicine_before_MCI_cluster, f)
f.close()

out=open(save_path_med+'medicine_stats.txt','w')
# save percentage
#medicine_names
for i in range(num_clusters):
    out.write('cluster_' + str(i + 1)+'\n')
    cluster_size = len(p_id_cluster_data.loc[p_id_cluster_data.Cluster == (i+1)])
    dic_cluster = medicine_before_MCI_cluster['cluster_'+str(i+1)]
    new_dic_cluster = {key: str((value / cluster_size)*100)[:5] for key, value in dic_cluster.items()}
    for m in medicine_names:
        out.write('%s\t%i\t%s\n' % (m,dic_cluster[m],new_dic_cluster[m]))

out.close()






