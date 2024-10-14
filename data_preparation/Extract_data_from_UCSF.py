
import pandas as pd
import numpy as np
from tqdm import tqdm
import scipy.io as sio
import scipy.sparse as ss

from function import *


# You need to change this path
diagnosis_table_path = './data/diagnosis_sample.csv'
procedure_table_path = './data/procedure_sample.csv'
drug_table_path = './data/medication_sample.csv'
demo_path = './data/demographics_sample.csv'

# You do not need to change the following path
icd9_phecode_mapping_path = './mapping/phecode_icd9_rolled.csv'
icd10_phecode_mapping_path = './mapping/phecode_icd10.csv'
ndc2rxnorm_mapping_path = './mapping/ndc2rxnorm.csv'
phecode_dic_path = './dic/phecode_definitions1.2.csv'

min_length = 2

#######################################################

# Step 1: Extract the patients who have both MCI and AD
# Input: diagnosis table (diagnosis_table_path). You don't need to select the records of patient in the cohort. Just input the original diagnosis table,
# and this code will select the patient (build cohort) according to the EC.

# If you input all records for patients in the cohort (1765 with MCI before AD), it is OK for running the code.
# Here, we do not require the progression time is longer than one year

print('Step 1: Extract the patients who have both MCI and AD')
Patient_has_MCI_and_AD = Get_PATID_having_MCI_and_AD(diangosis_table_path=diagnosis_table_path)
Patient_has_MCI_and_AD.to_csv('./Temp_data/Patient_ID_having_MCI_and_AD.csv')
del Patient_has_MCI_and_AD # reduce the memory

print('Step 2: Extract the patients whose data of AD onset is later than that of MCI, and the progressive year is longer than 365 days')
Patient_MCI_before_AD = Get_PATID_MCI_before_AD(diangosis_table_path=diagnosis_table_path,
                        PATID_having_MCI_AD_path = './Temp_data/Patient_ID_having_MCI_and_AD.csv')
Patient_MCI_before_AD.to_csv('./Temp_data/Patient_ID_MCI_before_AD.csv')
del Patient_MCI_before_AD # reduce the memory

###########################################################################################

print('Step 3: Extract all diagnosis, procedure and drug records from original tables with the list of patients from Step 2 (Time is limited as from MCI to AD')
Diag_Records_between_MCI_AD = Get_all_Diag_record_according_to_PATID(PATID_MCI_before_AD_path = './Temp_data/Patient_ID_MCI_before_AD.csv',
                                       diangosis_table_path = diagnosis_table_path)
Diag_Records_between_MCI_AD.to_csv('./Temp_data/diagnosis_tmp.csv')
del Diag_Records_between_MCI_AD # reduce the memory

Drug_Records_between_MCI_AD = Get_all_Drug_record_according_to_PATID(PATID_MCI_before_AD_path = './Temp_data/Patient_ID_MCI_before_AD.csv',
                                       drug_table_path = drug_table_path)
Drug_Records_between_MCI_AD.to_csv('./Temp_data/medication_tmp.csv')
del Drug_Records_between_MCI_AD # reduce the memory

Procedure_Records_between_MCI_AD = Get_all_Procedure_record_according_to_PATID(PATID_MCI_before_AD_path = './Temp_data/Patient_ID_MCI_before_AD.csv',
                                       procedure_table_path = procedure_table_path)
Procedure_Records_between_MCI_AD.to_csv('./Temp_data/procedure_tmp.csv')
del Procedure_Records_between_MCI_AD # reduce the memory

######################################################################################################################################
print('Step 4: Collect (Re-organize) all diagnosis, procedure and drug records from Step 3 by time')
Diag_collect_by_time = Collect_Diag_record_by_time(diangosis_table_path='./Temp_data/diagnosis_tmp.csv')
Diag_collect_by_time.to_csv('./Temp_data/diagnosis_step1.csv')
del Diag_collect_by_time

Drug_collect_by_time = Collect_Drug_record_by_time(drug_table_path='./Temp_data/medication_tmp.csv',
                                                   diangosis_table_path='./Temp_data/diagnosis_tmp.csv')
Drug_collect_by_time.to_csv('./Temp_data/medication_step1.csv')
del Drug_collect_by_time

Procedure_collect_by_time = Collect_Procedure_record_by_time(procedure_table_path='./Temp_data/procedure_tmp.csv',
                                                             diangosis_table_path='./Temp_data/diagnosis_tmp.csv')
Procedure_collect_by_time.to_csv('./Temp_data/procedure_step1.csv')
del Procedure_collect_by_time

########################################################################################################################################
print('Step 5.1: Map ICD code to Phecode')
Diag_from_ICD2Phecode = Map_from_ICD2Phecode(diangosis_table_path='./Temp_data/diagnosis_step1.csv',
                     icd9_phecode_mapping_path=icd9_phecode_mapping_path,
                     icd10_phecode_mapping_path=icd10_phecode_mapping_path)
Diag_from_ICD2Phecode.to_csv('./Temp_data/diagnosis_step2.csv')
del Diag_from_ICD2Phecode

print('Step 5.2: Intergarte NDC and RxNorm to RxNorm')
Drug_from_NDC2RxNorm = Map_from_NDC2RxNorm(drug_table_path='./Temp_data/medication_step1.csv',
                     ndc2rxnorm_mapping_path=ndc2rxnorm_mapping_path)
Drug_from_NDC2RxNorm.to_csv('./Temp_data/medication_step2.csv')
del Drug_from_NDC2RxNorm

########################################################################################################################################
print('Step 6: Prepare longitudinal diagnosis, procedure, and drug data')
temporal_Diagosis = Prepare_temporal_Diagosis(diangosis_table_path='./Temp_data/diagnosis_step2.csv')
temporal_Diagosis.to_csv('./Temp_data/diagnosis_step3.csv')
del temporal_Diagosis

temporal_Drug = Prepare_temporal_Drug(diangosis_table_path='./Temp_data/diagnosis_step3.csv',
                                                drug_table_path='./Temp_data/medication_step2.csv')
temporal_Drug.to_csv('./Temp_data/medication_step3.csv')
del temporal_Drug

temporal_Procedure = Prepare_temporal_Procedure(diangosis_table_path='./Temp_data/diagnosis_step3.csv',
                                                procedure_table_path='./Temp_data/procedure_step1.csv')
temporal_Procedure.to_csv('./Temp_data/procedure_step3.csv')
del temporal_Procedure

###############################################################################################################################
print('Step 7: Prepare dictionary for diagnosis, procedure, and drug data')
diag_code2name, diag_code2index = Prepare_Dic_for_Diag(diangosis_table_path='./Temp_data/diagnosis_step3.csv',
                                                       phecode_dic_path=phecode_dic_path)
np.save('./Temp_data/diag_code2name.npy', diag_code2name)
np.save('./Temp_data/diag_code2index.npy', diag_code2index)
del diag_code2name, diag_code2index

drug_code2index = Prepare_Dic_for_Drug(drug_table_path='./Temp_data/medication_step3.csv')
np.save('./Temp_data/drug_code2index.npy', drug_code2index)
del drug_code2index

proc_code2index = Prepare_Dic_for_Proc(procedure_table_path='./Temp_data/procedure_step3.csv')
np.save('./Temp_data/proc_code2index.npy', proc_code2index)
del proc_code2index

#######################################################################################################################################################
print('Step 8: Prepare diagnosis, procedure, and drug data for running model')
DATA_Diag, Patient_label_Diag = Prepare_Diagnosis_for_runnning_model(diag_code2index_path='./Temp_data/diag_code2index.npy',
                                     diangosis_table_path='./Temp_data/diagnosis_step3.csv',
                                     demo_path=demo_path,
                                     min_length = min_length)
np.save('./Temp_data/Data_diag.npy', DATA_Diag)
np.save('./Temp_data/Patient_label_diag.npy', Patient_label_Diag)
del DATA_Diag

DATA_Drug, Patient_label_Drug = Prepare_Drug_for_runnning_model(drug_code2index_path='./Temp_data/drug_code2index.npy',
                                     drug_table_path='./Temp_data/medication_step3.csv',
                                     demo_path=demo_path,
                                     min_length = min_length)
np.save('./Temp_data/Data_drug.npy', DATA_Drug)
np.save('./Temp_data/Patient_label_drug.npy', Patient_label_Drug)
del DATA_Drug

DATA_Proc, Patient_label_Proc = Prepare_Procedure_for_runnning_model(proc_code2index_path='./Temp_data/proc_code2index.npy',
                                     procedure_table_path='./Temp_data/procedure_step3.csv',
                                     demo_path=demo_path,
                                     min_length = min_length)
np.save('./Temp_data/Data_proc.npy', DATA_Proc)
np.save('./Temp_data/Patient_label_proc.npy', Patient_label_Proc)
del DATA_Proc

if np.sum(Patient_label_Diag - Patient_label_Drug)!=0:
    print('Warning: diagnosis and medication are not matched!')
if np.sum(Patient_label_Diag - Patient_label_Proc)!=0:
    print('Warning: diagnosis and procedure are not matched!')

##########################################################################################################################################################
print('Step 9: Get demogrphics for all selected patients and time sheet for clustering')
Demo, Time_sheet = Get_Demo_for_selective_patient(diangosis_table_path='./Temp_data/diagnosis_step3.csv',
                               demo_path=demo_path,
                               patient_list_path='./Temp_data/Patient_ID_MCI_before_AD.csv',
                               min_length=min_length)

Demo.to_csv('./Temp_data/demo_condition_minLength_'+str(min_length)+'_Age_Larger_than_50.csv')
np.save('./Temp_data/time_sheet_condition_minLength_'+str(min_length)+'_Age_Larger_than_50.npy', np.array(Time_sheet))

print('Well Done!')