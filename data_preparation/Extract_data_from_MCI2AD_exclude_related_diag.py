import pandas as pd
import numpy as np
# from tqdm import tqdm
# import scipy.io as sio
import scipy.sparse as ss
#from scipy.sparse import csr_matrix

temp_data_path='./Temp_data_exclude_dementia/'
temp_data_save_path='./Temp_data_excludeRelatedDiag/'

############## ####read previous data
data_diag=np.load(temp_data_path+'Data_diag.npy',allow_pickle=True).item().toarray()
data_drug=np.load(temp_data_path+'Data_drug.npy',allow_pickle=True).item().toarray()
data_proc=np.load(temp_data_path+'Data_proc.npy',allow_pickle=True).item().toarray()

diag_code2index=np.load(temp_data_path+'diag_code2index.npy',allow_pickle=True).tolist()
drug_code2index=np.load(temp_data_path+'drug_code2index.npy',allow_pickle=True).tolist()
procedure_code2index=np.load(temp_data_path+'procedure_code2index.npy',allow_pickle=True).tolist()

Patient_label=np.load(temp_data_path+'Patient_label_diag.npy',allow_pickle=True)

################ read removed code list
remove_diag_df=pd.read_csv('mci_ad_dementia_related.csv')
remove_diag_icd9_list=list(remove_diag_df.loc[remove_diag_df['Code_type']=='ICD-9']['Code'])
remove_diag_icd10_list=list(remove_diag_df.loc[remove_diag_df['Code_type']=='ICD-10']['Code'])

### read mapping
df_icd9_to_phecode = pd.read_csv('./mapping/phecode_icd9_rolled.csv', dtype=str)
df_icd10_to_phecode = pd.read_csv('./mapping/phecode_icd10.csv', dtype=str)

dic_icd9_to_phecode = dict(zip(list(df_icd9_to_phecode.ICD9), list(df_icd9_to_phecode.PheCode)))
dic_icd10_to_phecode = dict(zip(list(df_icd10_to_phecode.ICD10), list(df_icd10_to_phecode.PheCode)))
dic_icd10_to_phecode['G31.84']='292.2'

#### map ICD9/10 to phecode
phecode = []
icd9_more = []
icd10_more = []

for code in remove_diag_icd9_list:
    if code in dic_icd9_to_phecode:
        phecode.append(dic_icd9_to_phecode[code])
    elif code[0:-1] in dic_icd9_to_phecode:
        phecode.append(dic_icd9_to_phecode[code[0:-1]])
    else:
        codetmp = code.split('.')[0]
        if codetmp in dic_icd9_to_phecode:
            phecode.append(dic_icd9_to_phecode[codetmp])
        else:
            icd9_more.append(code)

for code in remove_diag_icd10_list:
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
                icd10_more.append(code)

phecode=list(set(phecode))
phecode.append('290.2')

#####
rm_dict={}
for i in phecode:
	if i in diag_code2index.keys():
		rm_dict[i]=diag_code2index[i]

rm_ind=list(rm_dict.values())

################# remove rows in data_diag
mask = np.ones(data_diag.shape[0], dtype=bool)
mask[rm_ind] = False
data_diag_new = data_diag[mask]

################# update diag_code2index
sorted_rm_dict = dict(sorted(rm_dict.items(), key=lambda item: item[1], reverse=False))
sorted_rm_ind = list(sorted_rm_dict.values())

diag_code2index_new = diag_code2index.copy()

for k in phecode:## remove phecode
	if k in diag_code2index_new.keys():
		del diag_code2index_new[k]

# update values
diag_code2index_new_new={}
for i in range(len(diag_code2index_new)):
	k=list(diag_code2index_new.keys())[i]
	v=list(diag_code2index_new.values())[i]
	diff=[v-ind for ind in sorted_rm_ind]
	pos_num=sum(1 for value in diff if value > 0)
	diag_code2index_new_new[k]=v-pos_num

################# update patient_label
# for one visit, if it's the sum of diagnosis (0/1) at one time point is 0, remove that visit
# if the patient at total <=2 visit, remove the patient
Patient_label_new=[]
patient_mask=[]#remove columns
p_new=0
for p in list(set(Patient_label)):
	p_ind=list(Patient_label==p)
	diag_sub=data_diag_new[:,p_ind]
	col_sum=diag_sub.sum(axis=0)
	valid_visit_num=list(col_sum>0).count(True)
	if valid_visit_num<2:#less than 2 visits, remove the patient
		patient_mask+=[False]*len(col_sum)
	else:# more than 2 visits, remove invalid visits
		patient_mask+=list(col_sum>0)
		Patient_label_new+=[p_new]*valid_visit_num
		p_new+=1


################# update data_drug, data_procedure
data_drug_new=data_drug[:,patient_mask]
data_proc_new=data_proc[:,patient_mask]
data_diag_new_new=data_diag_new[:,patient_mask]

################# update demo and Time files (used in clustering)
demo=pd.read_csv(temp_data_path+'demo_condition_minLength_2_Age_Larger_than_50.csv')
time=np.load(temp_data_path+'time_sheet_condition_minLength_2_Age_Larger_than_50.npy')
patient_keep=list(set(Patient_label[patient_mask]))
demo_new=demo.loc[demo['Unnamed: 0'].isin(patient_keep)]
demo_new.drop(columns='Unnamed: 0',inplace=True)

time_new=time[patient_mask]

################ save updated results
np.save(temp_data_save_path+'Data_diag.npy', ss.coo_matrix(data_diag_new_new))
np.save(temp_data_save_path+'Data_drug.npy', ss.coo_matrix(data_drug_new))
np.save(temp_data_save_path+'Data_proc.npy', ss.coo_matrix(data_proc_new))

np.save(temp_data_save_path+'diag_code2index.npy', diag_code2index_new_new)

np.save(temp_data_save_path+'Patient_label_diag.npy', Patient_label_new)
np.save(temp_data_save_path+'Patient_label_drug.npy', Patient_label_new)
np.save(temp_data_save_path+'Patient_label_proc.npy', Patient_label_new)

demo_new.to_csv(temp_data_save_path+'demo_condition_minLength_2_Age_Larger_than_50.csv')
np.save(temp_data_save_path+'time_sheet_condition_minLength_2_Age_Larger_than_50.npy',time_new)








