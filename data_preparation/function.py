
import pandas as pd
import numpy as np
from tqdm import tqdm
import scipy.io as sio
import scipy.sparse as ss
from datetime import datetime, timedelta

def Get_PATID_having_MCI_and_AD(diangosis_table_path):

    ## This is function is used to extract the patients who have both MCI and AD

    df_diagnosis = pd.read_csv(diangosis_table_path, dtype=str)

    PATID_AD_MCI = []

    tmp_MCI = df_diagnosis[df_diagnosis['Value'].isin(['331.83', '294.9', 'G31.84', 'F09'])]
    PATID_MCI = list(tmp_MCI['PatientID'].unique())

    tmp_AD = df_diagnosis[df_diagnosis['Value'].isin(['331', 'G30', 'G30.0', 'G30.1', 'G30.8', 'G30.9'])]
    PATID_AD = list(tmp_AD['PatientID'].unique())

    PATID_AD_MCI = list(set(PATID_MCI).intersection(set(PATID_AD)))

    dataframe = pd.DataFrame.from_dict({'PATID': pd.Series(PATID_AD_MCI)})

    return dataframe

def Get_PATID_MCI_before_AD(diangosis_table_path, PATID_having_MCI_AD_path):

    ## This function is used to extract the patients who have both MCI and AD, but the data of AD onset is later than that of MCI

    df_diagnosis = pd.read_csv(diangosis_table_path, dtype=str)
    Patients = pd.read_csv(PATID_having_MCI_AD_path, index_col=0, dtype=str)

    PATID = []
    MCI_TIME = []
    AD_TIME = []

    for patient in tqdm(Patients['PATID']):
        Records_one_patient = df_diagnosis.loc[df_diagnosis['PatientID'] == patient]

        Records_one_patient_MCI = Records_one_patient[Records_one_patient['Value'].isin(['331.83', '294.9', 'G31.84', 'F09'])]
        time_MCI = pd.to_datetime(Records_one_patient_MCI['StartDate'].min())

        Records_one_patient_AD = Records_one_patient[Records_one_patient['Value'].isin(['331', 'G30', 'G30.0', 'G30.1', 'G30.8', 'G30.9'])]
        time_AD = pd.to_datetime(Records_one_patient_AD['StartDate'].min())

        if time_MCI < time_AD:
            PATID.append(patient)
            MCI_TIME.append(time_MCI)
            AD_TIME.append(time_AD)

    dataframe = pd.DataFrame.from_dict(
        {'PATID': pd.Series(PATID),
        'MCI_TIME': pd.Series(MCI_TIME),
        'AD_TIME': pd.Series(AD_TIME)
         })

    return dataframe

def Get_all_Diag_record_according_to_PATID(PATID_MCI_before_AD_path, diangosis_table_path):
    Patients = pd.read_csv(PATID_MCI_before_AD_path, index_col=0, dtype=str)
    df_diagnosis = pd.read_csv(diangosis_table_path, dtype=str)

    df_diag_tmp = pd.DataFrame()

    for patient in tqdm(list(Patients['PATID'])):
        start_time = pd.to_datetime(np.array(Patients.loc[Patients['PATID'] == patient]['MCI_TIME'])[0])
        end_time = pd.to_datetime(np.array(Patients.loc[Patients['PATID'] == patient]['AD_TIME'])[0])

        Records_one_patient = df_diagnosis.loc[df_diagnosis['PatientID'] == patient]
        Records_one_patient_tmp = Records_one_patient.loc[(pd.to_datetime(Records_one_patient['StartDate'])>=start_time)
                                                         & (pd.to_datetime(Records_one_patient['StartDate'])<=end_time)]
        df_diag_tmp = pd.concat([df_diag_tmp, Records_one_patient_tmp], axis=0, ignore_index=True)

    return df_diag_tmp

def Get_all_Procedure_record_according_to_PATID(PATID_MCI_before_AD_path, procedure_table_path):
    Patients = pd.read_csv(PATID_MCI_before_AD_path, index_col=0, dtype=str)
    df_procedure = pd.read_csv(procedure_table_path, dtype=str)

    df_procedure_tmp = pd.DataFrame()

    for patient in tqdm(list(Patients['PATID'])):
        start_time = pd.to_datetime(np.array(Patients.loc[Patients['PATID'] == patient]['MCI_TIME'])[0])
        end_time = pd.to_datetime(np.array(Patients.loc[Patients['PATID'] == patient]['AD_TIME'])[0])

        Records_one_patient = df_procedure.loc[df_procedure['PatientID'] == patient]
        Records_one_patient_tmp = Records_one_patient.loc[(pd.to_datetime(Records_one_patient['OrderDate'])>=start_time)
                                                         & (pd.to_datetime(Records_one_patient['OrderDate'])<=end_time)]
        Records_one_patient_tmp = Records_one_patient_tmp.loc[Records_one_patient_tmp['CodeSet']=='CPT'] # Only use CPT code
        df_procedure_tmp = pd.concat([df_procedure_tmp, Records_one_patient_tmp], axis=0, ignore_index=True)

    return df_procedure_tmp

def Get_all_Drug_record_according_to_PATID(PATID_MCI_before_AD_path, drug_table_path):
    Patients = pd.read_csv(PATID_MCI_before_AD_path, index_col=0, dtype=str)
    df_drug = pd.read_csv(drug_table_path, dtype=str)

    df_drug_tmp = pd.DataFrame()

    for patient in tqdm(list(Patients['PATID'])):
        start_time = pd.to_datetime(np.array(Patients.loc[Patients['PATID'] == patient]['MCI_TIME'])[0])
        end_time = pd.to_datetime(np.array(Patients.loc[Patients['PATID'] == patient]['AD_TIME'])[0])

        Records_one_patient = df_drug.loc[df_drug['PatientID'] == patient]
        Records_one_patient_tmp = Records_one_patient.loc[(pd.to_datetime(Records_one_patient['OrderDate'])>=start_time)
                                                         & (pd.to_datetime(Records_one_patient['OrderDate'])<=end_time)]
        df_drug_tmp = pd.concat([df_drug_tmp, Records_one_patient_tmp], axis=0, ignore_index=True)

    return df_drug_tmp

def Collect_Diag_record_by_time(diangosis_table_path):
    df_diagnosis = pd.read_csv(diangosis_table_path, dtype=str, index_col=0)
    All_patients = df_diagnosis['PatientID'].unique()

    PATID = []
    ADMIT_DATE = []
    ICD = []

    for patient in tqdm(All_patients):
        one_patient_all_records_diagnosis = df_diagnosis.loc[df_diagnosis['PatientID'] == patient]

        one_patient_unque_admis = list(pd.to_datetime(np.sort(pd.to_datetime(one_patient_all_records_diagnosis['StartDate'].unique())))) # from old to new
        for admit in one_patient_unque_admis:
            time = str(admit.year)+'/'+str(admit.month)+'/'+str(admit.day)
            one_patient_one_admit_recode = one_patient_all_records_diagnosis.loc[one_patient_all_records_diagnosis['StartDate'] == time]
            icd = list(one_patient_one_admit_recode['Value'])

            PATID.append(patient)
            ADMIT_DATE.append(admit)
            ICD.append(' '.join(icd))

    dataframe = pd.DataFrame.from_dict(
        {'PATID': pd.Series(PATID),
         'ADMIT_DATE':pd.Series(ADMIT_DATE),
         'ICD':pd.Series(ICD)
         })

    return dataframe

def Collect_Procedure_record_by_time(procedure_table_path, diangosis_table_path):
    df_procedure = pd.read_csv(procedure_table_path, dtype=str, index_col=0)

    tmp = pd.read_csv(diangosis_table_path, dtype=str, index_col=0)
    All_patients = tmp['PatientID'].unique()

    PATID = []
    ADMIT_DATE = []
    CPT = []

    for patient in tqdm(All_patients):
        one_patient_all_records_procedure = df_procedure.loc[df_procedure['PatientID'] == patient]
        one_patient_unque_admis = np.sort(one_patient_all_records_procedure['OrderDate'].unique()) # from old to new
        for admit in one_patient_unque_admis:
            one_patient_one_admit_recode = one_patient_all_records_procedure.loc[one_patient_all_records_procedure['OrderDate'] == admit]
            code = list(one_patient_one_admit_recode['Code'])

            PATID.append(patient)
            ADMIT_DATE.append(admit)
            CPT.append(' '.join(code))

    dataframe = pd.DataFrame.from_dict(
        {'PATID': pd.Series(PATID),
         'ADMIT_DATE':pd.Series(ADMIT_DATE),
         'CPT':pd.Series(CPT)
         })

    return dataframe

def Collect_Drug_record_by_time(drug_table_path, diangosis_table_path):
    df_drug = pd.read_csv(drug_table_path, dtype=str, index_col=0)

    tmp = pd.read_csv(diangosis_table_path, dtype=str, index_col=0)
    All_patients = tmp['PatientID'].unique()

    PATID = []
    ADMIT_DATE = []
    NDC = []
    RxNorm = []

    for patient in tqdm(All_patients):
        one_patient_all_records_drug = df_drug.loc[df_drug['PatientID'] == patient]
        one_patient_unque_admis = list(pd.to_datetime(np.sort(pd.to_datetime(one_patient_all_records_drug['OrderDate'].unique())))) # from old to new
        for admit in one_patient_unque_admis:
            time = str(admit.year) + '/' + str(admit.month) + '/' + str(admit.day)
            one_patient_one_admit_recode = one_patient_all_records_drug.loc[one_patient_all_records_drug['OrderDate'] == time]
            NDC_records = one_patient_one_admit_recode.loc[one_patient_one_admit_recode['Type'] == 'NDC']
            RxNorm_records = one_patient_one_admit_recode.loc[one_patient_one_admit_recode['Type'] == 'RxNorm']

            PATID.append(patient)
            ADMIT_DATE.append(admit)
            NDC.append(' '.join(list(NDC_records['Code'])))
            RxNorm.append(' '.join(list(RxNorm_records['Code'])))

    dataframe = pd.DataFrame.from_dict(
        {'PATID': pd.Series(PATID),
         'ADMIT_DATE':pd.Series(ADMIT_DATE),
         'NDC':pd.Series(NDC),
         'RxNorm': pd.Series(RxNorm)
         })

    return dataframe

def Map_from_ICD2Phecode(diangosis_table_path, icd9_phecode_mapping_path, icd10_phecode_mapping_path):
    df_diagnosis = pd.read_csv(diangosis_table_path, dtype=str, index_col=0)
    df_icd9_to_phecode = pd.read_csv(icd9_phecode_mapping_path, dtype=str)
    df_icd10_to_phecode = pd.read_csv(icd10_phecode_mapping_path, dtype=str)
    Num_records = df_diagnosis.shape[0]

    dic_icd9_to_phecode = dict(zip(list(df_icd9_to_phecode.ICD9), list(df_icd9_to_phecode.PheCode)))
    dic_icd10_to_phecode = dict(zip(list(df_icd10_to_phecode.ICD10), list(df_icd10_to_phecode.PheCode)))
    dic_icd10_to_phecode['G31.84']='292.2'

    PATID = []
    ADMIT_DATE = []
    PHECODE = []

    for i in tqdm(range(Num_records)):
        one_record = df_diagnosis.iloc[i,:]
        patient = one_record['PATID']
        admit_time = one_record['ADMIT_DATE']
        if isinstance(one_record['ICD'], str):
            icd = one_record['ICD'].split(' ')
        else:
            icd = []

        phecode = []

        for code in icd:
            try:
                code = str(float(code))
            except:
                code = code

            if code in dic_icd9_to_phecode:
                phecode.append(dic_icd9_to_phecode[code])
                continue
            else:
                if code in dic_icd10_to_phecode:
                    phecode.append(dic_icd10_to_phecode[code])
                    continue
                elif code[0:5] in dic_icd10_to_phecode:
                    phecode.append(dic_icd10_to_phecode[code[0:5]])
                    continue
                elif code[0:3] in dic_icd10_to_phecode:
                    phecode.append(dic_icd10_to_phecode[code[0:3]])
                    continue
                else:
                    phecode.append(' ')



        PATID.append(patient)
        ADMIT_DATE.append(admit_time)
        phecode = list(set(phecode))

        if len(phecode)>0:
            PHECODE.append(' '.join(phecode))
        else:
            PHECODE.append(' ')


    dataframe = pd.DataFrame.from_dict(
        {'PATID': pd.Series(PATID),
         'ADMIT_DATE':pd.Series(ADMIT_DATE),
         'PHECODE':pd.Series(PHECODE)
         })

    return dataframe

def Map_from_NDC2RxNorm(drug_table_path, ndc2rxnorm_mapping_path):
    df_drug = pd.read_csv(drug_table_path, dtype=str, index_col=0)
    df_ndc2rxnorm = pd.read_csv(ndc2rxnorm_mapping_path, dtype=str, index_col=0)
    Num_records = df_drug.shape[0]

    dic_ndc_to_rxcui = dict(zip(list(df_ndc2rxnorm['Code']), list(df_ndc2rxnorm['RXCUI'])))

    PATID = []
    ADMIT_DATE = []
    RXNORM = []

    for i in tqdm(range(Num_records)):
        one_record = df_drug.iloc[i,:]
        patient = one_record['PATID']
        admit_time = one_record['ADMIT_DATE']

        original_rxnorm = one_record['RxNorm']
        if isinstance(original_rxnorm, str):
            original_rxnorm = original_rxnorm.split(' ')
        else:
            original_rxnorm = []

        if isinstance(one_record['NDC'], str):
            ndc = one_record['NDC'].split(' ')
        else:
            ndc = []

        for code in ndc:
            tmp = dic_ndc_to_rxcui[code]
            if isinstance(tmp, str):
                original_rxnorm.append(tmp)

        PATID.append(patient)
        ADMIT_DATE.append(admit_time)
        new_rxnorm = list(set(original_rxnorm))

        if len(new_rxnorm)>0:
            RXNORM.append(' '.join(new_rxnorm))
        else:
            RXNORM.append(' ')


    dataframe = pd.DataFrame.from_dict(
        {'PATID': pd.Series(PATID),
         'ADMIT_DATE':pd.Series(ADMIT_DATE),
         'RxNorm':pd.Series(RXNORM)
         })

    return dataframe

def Prepare_temporal_Diagosis(diangosis_table_path):
    df_diagnosis = pd.read_csv(diangosis_table_path, dtype=str, index_col=0)

    All_patients = df_diagnosis['PATID'].unique()

    PATID = []
    TIME_Index = []
    START_TIME = []
    END_TIME = []
    PHECODE = []

    for patient in tqdm(list(All_patients)):
        flag = 1
        one_patient_all_records_diagnosis = df_diagnosis.loc[df_diagnosis['PATID'] == patient]

        min_time = pd.to_datetime(np.min(one_patient_all_records_diagnosis['ADMIT_DATE'].unique()))
        max_time = pd.to_datetime(np.max(one_patient_all_records_diagnosis['ADMIT_DATE'].unique()))

        start_year = min_time.year
        start_month = min_time.month

        if start_month == 12:
            end_year = start_year+1
            end_month = 3
        elif start_month == 11:
            end_year = start_year+1
            end_month = 2
        elif start_month == 10:
            end_year = start_year+1
            end_month = 1
        else:
            end_year = start_year
            end_month = start_month+3

        end_time = pd.to_datetime(datetime(end_year, end_month, 1)) - timedelta(days=1)
        start_time = min_time

        while start_time < max_time:
            tmp = one_patient_all_records_diagnosis.loc[(pd.to_datetime(one_patient_all_records_diagnosis['ADMIT_DATE']) >= start_time)
                                                       & (pd.to_datetime(one_patient_all_records_diagnosis['ADMIT_DATE']) <= end_time)]

            phecode = []
            for i in range(tmp.shape[0]):
                if isinstance(tmp.iloc[i,:]['PHECODE'], str):
                    phecode_tmp = tmp.iloc[i,:]['PHECODE'].split()
                    phecode_tmp = [x for x in phecode_tmp if x != 'nan']
                    phecode = phecode + phecode_tmp

            if len(phecode)>0:
                PATID.append(patient)
                PHECODE.append(' '.join(phecode))
                TIME_Index.append(flag)
                START_TIME.append(start_time)
                END_TIME.append(end_time)
                flag += 1

            if end_time.month == 12:
                start_year = end_time.year + 1
                start_month = 1
            else:
                start_year = end_time.year
                start_month = end_time.month+1

            if start_month == 12:
                end_year = start_year + 1
                end_month = 3
            elif start_month == 11:
                end_year = start_year + 1
                end_month = 2
            elif start_month == 10:
                end_year = start_year + 1
                end_month = 1
            else:
                end_year = start_year
                end_month = start_month + 3

            end_time = pd.to_datetime(datetime(end_year, end_month, 1)) - timedelta(days=1)
            start_time = pd.to_datetime(datetime(start_year, start_month, 1))

    dataframe = pd.DataFrame.from_dict(
        {'PATID': pd.Series(PATID),
         'TIME_Index':pd.Series(TIME_Index),
         'START_TIME':pd.Series(START_TIME),
         'END_TIME':pd.Series(END_TIME),
         'PHECODE':pd.Series(PHECODE)
         })

    return dataframe

def Prepare_temporal_Procedure(diangosis_table_path, procedure_table_path):
    df_diagnosis = pd.read_csv(diangosis_table_path, index_col=0, dtype=str)
    df_procedure = pd.read_csv(procedure_table_path, index_col=0, dtype=str)

    subjects = df_diagnosis['PATID'].unique()

    PATID = []
    TIME_Index = []
    START_TIME = []
    END_TIME = []
    CPT = []

    for subject_id in tqdm(subjects):
        flag = 1
        data_one_patient = df_diagnosis.loc[df_diagnosis['PATID'] == subject_id]
        unique_admissions = np.sort(data_one_patient['TIME_Index'].unique())

        data_one_patient_procedure = df_procedure.loc[df_procedure['PATID'] == subject_id]

        for admission_ID in unique_admissions:
            data_one_patient_one_admission = data_one_patient.loc[data_one_patient['TIME_Index'] == admission_ID]
            start_time = pd.to_datetime(data_one_patient_one_admission.iloc[0]['START_TIME'])
            end_time = pd.to_datetime(data_one_patient_one_admission.iloc[0]['END_TIME'])

            data_one_patient_one_admission_procedure = data_one_patient_procedure.loc[(pd.to_datetime(data_one_patient_procedure['ADMIT_DATE']) >= start_time) &
                                                                                     (pd.to_datetime(data_one_patient_procedure['ADMIT_DATE']) <= end_time)]

            cpt = []
            for i in range(data_one_patient_one_admission_procedure.shape[0]):
                if isinstance(data_one_patient_one_admission_procedure.iloc[i]['CPT'], str):
                    codes = data_one_patient_one_admission_procedure.iloc[i]['CPT'].split(' ')
                    codes = [x for x in codes if x != 'nan']

                    cpt = cpt+codes

            PATID.append(subject_id)
            CPT.append(' '.join(cpt))
            TIME_Index.append(flag)
            START_TIME.append(start_time)
            END_TIME.append(end_time)
            flag += 1

    dataframe = pd.DataFrame.from_dict(
        {'PATID': pd.Series(PATID),
         'TIME_Index':pd.Series(TIME_Index),
         'START_TIME':pd.Series(START_TIME),
         'END_TIME':pd.Series(END_TIME),
         'CPT':pd.Series(CPT)
         })

    return dataframe

def Prepare_temporal_Drug(diangosis_table_path, drug_table_path):
    df_diagnosis = pd.read_csv(diangosis_table_path, index_col=0, dtype=str)
    df_drug = pd.read_csv(drug_table_path, index_col=0, dtype=str)

    subjects = df_diagnosis['PATID'].unique()

    PATID = []
    TIME_Index = []
    START_TIME = []
    END_TIME = []
    RXNORM = []

    for subject_id in tqdm(subjects):
        flag = 1
        data_one_patient = df_diagnosis.loc[df_diagnosis['PATID'] == subject_id]
        unique_admissions = np.sort(data_one_patient['TIME_Index'].unique())

        data_one_patient_drug = df_drug.loc[df_drug['PATID'] == subject_id]

        for admission_ID in unique_admissions:
            data_one_patient_one_admission = data_one_patient.loc[data_one_patient['TIME_Index'] == admission_ID]
            start_time = pd.to_datetime(data_one_patient_one_admission.iloc[0]['START_TIME'])
            end_time = pd.to_datetime(data_one_patient_one_admission.iloc[0]['END_TIME'])

            data_one_patient_one_admission_medication = data_one_patient_drug.loc[(pd.to_datetime(data_one_patient_drug['ADMIT_DATE']) >= start_time) &
                                                                                     (pd.to_datetime(data_one_patient_drug['ADMIT_DATE']) <= end_time)]

            rxnorm = []
            for i in range(data_one_patient_one_admission_medication.shape[0]):
                if isinstance(data_one_patient_one_admission_medication.iloc[i]['RxNorm'], str):
                    codes = data_one_patient_one_admission_medication.iloc[i]['RxNorm'].split(' ')
                    codes = [x for x in codes if x != '']

                    rxnorm = rxnorm+codes

            PATID.append(subject_id)
            RXNORM.append(' '.join(rxnorm))
            TIME_Index.append(flag)
            START_TIME.append(start_time)
            END_TIME.append(end_time)
            flag += 1

    dataframe = pd.DataFrame.from_dict(
        {'PATID': pd.Series(PATID),
         'TIME_Index':pd.Series(TIME_Index),
         'START_TIME':pd.Series(START_TIME),
         'END_TIME':pd.Series(END_TIME),
         'RXNORM':pd.Series(RXNORM)
         })

    return dataframe

def Prepare_Dic_for_Diag(diangosis_table_path, phecode_dic_path):

    df_diagnosis = pd.read_csv(diangosis_table_path, dtype=str, index_col=0)
    phecode_definition = pd.read_csv(phecode_dic_path, dtype=str)
    phecode_Dic = dict(zip(list(phecode_definition.iloc[:, 0]), list(phecode_definition.iloc[:, 1])))

    diag_code2name = {}
    diag_code2index = {}
    flag = 0

    for i in tqdm(range(df_diagnosis.shape[0])):
        one_record = df_diagnosis.iloc[i, :]
        codes = one_record['PHECODE'].split(' ')
        for code in codes:
            if code in diag_code2name:
                continue
            else:
                if code in phecode_Dic:
                    mean = phecode_Dic[code]
                else:
                    mean = 'NF'

                diag_code2name[code] = mean
                diag_code2index[code] = flag
                flag += 1

    return diag_code2name, diag_code2index

def Prepare_Dic_for_Proc(procedure_table_path):

    df_procedure = pd.read_csv(procedure_table_path, dtype=str, index_col=0)

    proc_code2index = {}
    flag = 0

    for i in tqdm(range(df_procedure.shape[0])):
        one_record = df_procedure.iloc[i, :]
        if isinstance(one_record['CPT'], np.float):
            continue
        codes = one_record['CPT'].split(' ')
        for code in codes:
            if code in proc_code2index:
                continue
            else:
                proc_code2index[code] = flag
                flag += 1

    return proc_code2index

def Prepare_Dic_for_Drug(drug_table_path):

    df_drug = pd.read_csv(drug_table_path, dtype=str, index_col=0)

    drug_code2index = {}
    flag = 0

    for i in tqdm(range(df_drug.shape[0])):
        one_record = df_drug.iloc[i, :]
        if isinstance(one_record['RXNORM'], np.float):
            continue
        codes = one_record['RXNORM'].split(' ')
        for code in codes:
            if code in drug_code2index:
                continue
            else:
                drug_code2index[code] = flag
                flag += 1

    return drug_code2index

def Prepare_Diagnosis_for_runnning_model(diag_code2index_path, diangosis_table_path, demo_path, min_length):

    diag_code2index = np.load(diag_code2index_path, allow_pickle=True).item()
    df_diagnosis = pd.read_csv(diangosis_table_path, index_col=0, dtype=str)
    df_demo = pd.read_csv(demo_path, dtype=str)

    subjects = df_diagnosis['PATID'].unique()
    Total_unique_code = len(diag_code2index)

    flag = 0
    for subject_id in tqdm(subjects):
        data_one_patient = df_diagnosis.loc[df_diagnosis['PATID'] == subject_id]

        unique_admissions = data_one_patient['TIME_Index'].unique()

        if len(unique_admissions) >= min_length:
            tmp = pd.to_datetime(np.array(data_one_patient.loc[data_one_patient['TIME_Index'] == '1']['START_TIME']))\
                  - pd.to_datetime(np.array(df_demo.loc[df_demo['PatientID'] == subject_id]['BirthDate'])[0])

            if int(tmp.days.values/365) >= 50:
                flag += len(unique_admissions)

    DATA = np.zeros([Total_unique_code, flag])
    Patient_label = np.zeros(flag)

    flag = 0
    patient_index = -1

    for subject_id in tqdm(subjects):

        data_one_patient = df_diagnosis.loc[df_diagnosis['PATID'] == subject_id]

        unique_admissions = np.sort(data_one_patient['TIME_Index'].unique())

        if len(unique_admissions) >= min_length:

            tmp = pd.to_datetime(np.array(data_one_patient.loc[data_one_patient['TIME_Index'] == '1']['START_TIME']))\
                  - pd.to_datetime(np.array(df_demo.loc[df_demo['PatientID'] == subject_id]['BirthDate'])[0])

            if int(tmp.days.values/365) >= 50:
                patient_index += 1

                for admission_ID in unique_admissions:
                    data_one_patient_one_admission = data_one_patient.loc[data_one_patient['TIME_Index'] == admission_ID]

                    codes = data_one_patient_one_admission.iloc[0]['PHECODE'].split(' ')

                    Patient_label[flag] = patient_index
                    for code in codes:
                        DATA[diag_code2index[code], flag] += 1
                    flag += 1

    DATA_Diagnosis = ss.coo_matrix(DATA)

    return DATA_Diagnosis, Patient_label

def Prepare_Procedure_for_runnning_model(proc_code2index_path, procedure_table_path, demo_path, min_length):

    proc_code2index = np.load(proc_code2index_path, allow_pickle=True).item()
    df_procedure = pd.read_csv(procedure_table_path, index_col=0, dtype=str)
    df_demo = pd.read_csv(demo_path, dtype=str)

    subjects = df_procedure['PATID'].unique()
    Total_unique_code = len(proc_code2index)

    flag = 0
    for subject_id in tqdm(subjects):
        data_one_patient = df_procedure.loc[df_procedure['PATID'] == subject_id]

        unique_admissions = data_one_patient['TIME_Index'].unique()

        if len(unique_admissions) >= min_length:
            tmp = pd.to_datetime(np.array(data_one_patient.loc[data_one_patient['TIME_Index'] == '1']['START_TIME']))\
                  - pd.to_datetime(np.array(df_demo.loc[df_demo['PatientID'] == subject_id]['BirthDate'])[0])

            if int(tmp.days.values/365) >= 50:
                flag += len(unique_admissions)

    DATA = np.zeros([Total_unique_code, flag])
    Patient_label = np.zeros(flag)

    flag = 0
    patient_index = -1

    for subject_id in tqdm(subjects):

        data_one_patient = df_procedure.loc[df_procedure['PATID'] == subject_id]

        unique_admissions = np.sort(data_one_patient['TIME_Index'].unique())

        if len(unique_admissions) >= min_length:

            tmp = pd.to_datetime(
                np.array(data_one_patient.loc[data_one_patient['TIME_Index'] == '1']['START_TIME'])) - pd.to_datetime(
                np.array(df_demo.loc[df_demo['PatientID'] == subject_id]['BirthDate'])[0])

            if int(tmp.days.values/365) >= 50:
                patient_index += 1

                for admission_ID in unique_admissions:
                    data_one_patient_one_admission = data_one_patient.loc[data_one_patient['TIME_Index'] == admission_ID]

                    if isinstance(data_one_patient_one_admission.iloc[0]['CPT'], np.float):
                        codes = []
                    else:
                        codes = data_one_patient_one_admission.iloc[0]['CPT'].split(' ')

                    Patient_label[flag] = patient_index
                    for code in codes:
                        DATA[proc_code2index[code], flag] += 1
                    flag += 1

    DATA_Procedure = ss.coo_matrix(DATA)

    return DATA_Procedure, Patient_label

def Prepare_Drug_for_runnning_model(drug_code2index_path, drug_table_path, demo_path, min_length):

    drug_code2index = np.load(drug_code2index_path, allow_pickle=True).item()
    df_drug = pd.read_csv(drug_table_path, index_col=0, dtype=str)
    df_demo = pd.read_csv(demo_path, dtype=str)

    subjects = df_drug['PATID'].unique()
    Total_unique_code = len(drug_code2index)

    flag = 0
    for subject_id in tqdm(subjects):
        data_one_patient = df_drug.loc[df_drug['PATID'] == subject_id]

        unique_admissions = data_one_patient['TIME_Index'].unique()

        if len(unique_admissions) >= min_length:
            tmp = pd.to_datetime(np.array(data_one_patient.loc[data_one_patient['TIME_Index'] == '1']['START_TIME']))\
                  - pd.to_datetime(np.array(df_demo.loc[df_demo['PatientID'] == subject_id]['BirthDate'])[0])

            if int(tmp.days.values/365) >= 50:
                flag += len(unique_admissions)

    DATA = np.zeros([Total_unique_code, flag])
    Patient_label = np.zeros(flag)

    flag = 0
    patient_index = -1

    for subject_id in tqdm(subjects):

        data_one_patient = df_drug.loc[df_drug['PATID'] == subject_id]

        unique_admissions = np.sort(data_one_patient['TIME_Index'].unique())

        if len(unique_admissions) >= min_length:

            tmp = pd.to_datetime(
                np.array(data_one_patient.loc[data_one_patient['TIME_Index'] == '1']['START_TIME'])) - pd.to_datetime(
                np.array(df_demo.loc[df_demo['PatientID'] == subject_id]['BirthDate'])[0])

            if int(tmp.days.values/365) >= 50:
                patient_index += 1

                for admission_ID in unique_admissions:
                    data_one_patient_one_admission = data_one_patient.loc[data_one_patient['TIME_Index'] == admission_ID]

                    if isinstance(data_one_patient_one_admission.iloc[0]['RXNORM'], np.float):
                        codes = []
                    else:
                        codes = data_one_patient_one_admission.iloc[0]['RXNORM'].split(' ')

                    Patient_label[flag] = patient_index
                    for code in codes:
                        DATA[drug_code2index[code], flag] += 1
                    flag += 1

    DATA_Drug = ss.coo_matrix(DATA)

    return DATA_Drug, Patient_label

def Get_Demo_for_selective_patient(diangosis_table_path, demo_path, patient_list_path, min_length):
    df_diagnosis = pd.read_csv(diangosis_table_path, dtype=str, index_col=0)
    df_demo = pd.read_csv(demo_path, dtype=str)

    subjects = df_diagnosis['PATID'].unique()
    df_demo = df_demo[df_demo['PatientID'].isin(subjects)]
    df_time = pd.read_csv(patient_list_path, dtype=str, index_col=0)

    flag = -1

    Patient_index = []
    PATID = []
    BIRTH_DATA = []
    SEX = []
    RACE = []
    MCI_AGE = []
    Time_sheet = []
    MCI2AD_time = []
    Num_visit = []

    for subject_id in tqdm(subjects):
        data_one_patient = df_diagnosis.loc[df_diagnosis['PATID'] == subject_id]

        unique_admissions = data_one_patient['TIME_Index'].unique()

        if len(unique_admissions) >= min_length:
            tmp = pd.to_datetime(np.array(data_one_patient.loc[data_one_patient['TIME_Index'] == '1']['START_TIME'])) - \
                  pd.to_datetime(np.array(df_demo.loc[df_demo['PatientID'] == subject_id]['BirthDate'])[0])
            if int(tmp.days.values/365) >= 50:
                flag += 1

                PATID.append(subject_id)
                Patient_index.append(flag)
                BIRTH_DATA.append(pd.to_datetime(np.array(df_demo.loc[df_demo['PatientID'] == subject_id]['BirthDate'])[0]))
                SEX.append(np.array(df_demo.loc[df_demo['PatientID'] == subject_id]['Sex'])[0])
                RACE.append(np.array(df_demo.loc[df_demo['PatientID'] == subject_id]['FirstRace'])[0])
                tmp = pd.to_datetime(np.array(data_one_patient.loc[data_one_patient['TIME_Index'] == '1']['START_TIME'])) - pd.to_datetime(np.array(df_demo.loc[df_demo['PatientID'] == subject_id]['BirthDate'])[0])
                MCI_AGE.append(int(tmp.days.values/365))

                tmp = pd.to_datetime(np.array(df_time.loc[df_time['PATID'] == subject_id]['AD_TIME'])) - pd.to_datetime(np.array(df_time.loc[df_time['PATID'] == subject_id]['MCI_TIME']))
                MCI2AD_time.append(int(tmp.days.values))

                Num_visit.append(len(unique_admissions))

                for admission_ID in unique_admissions:
                    data_one_patient_one_admission = data_one_patient.loc[data_one_patient['TIME_Index'] == admission_ID]
                    if admission_ID == '1':
                        Time_sheet.append(0)
                        initial_year = np.array(pd.to_datetime(data_one_patient_one_admission['END_TIME']).dt.year)[0]
                        initial_month = np.array(pd.to_datetime(data_one_patient_one_admission['END_TIME']).dt.month)[0]
                    else:
                        time_year = np.array(pd.to_datetime(data_one_patient_one_admission['END_TIME']).dt.year)[0]
                        time_month = np.array(pd.to_datetime(data_one_patient_one_admission['END_TIME']).dt.month)[0]

                        if initial_year == time_year:
                            difference = time_month - initial_month
                        else:
                            difference = 12*(time_year-initial_year-1)+(12-initial_month)+time_month

                        Time_sheet.append(difference)


    dataframe = pd.DataFrame.from_dict(
        {'PATID': pd.Series(PATID),
         'Patient_Index':pd.Series(Patient_index),
         'BIRTH_DATA':pd.Series(BIRTH_DATA),
         'SEX':pd.Series(SEX),
         'RACE':pd.Series(RACE),
         'MCIAGE':pd.Series(MCI_AGE),
         'MCI2AD_time':pd.Series(MCI2AD_time),
         'Num_visit':pd.Series(Num_visit)
         })

    return dataframe, Time_sheet

