import pandas as pd
import numpy as np
import sys
import os

###
model_path=sys.argv[1]#'./DMTM/python_code/trained_model_exclude_related_diag/topic30/'
save_path=sys.argv[2]#'./results/exclude_related_diag/'
ref_path=sys.argv[3]#'./data_preparation/mapping/'
top_N=int(sys.argv[4])#10#top 5 or 10

ref_diag=pd.read_csv(ref_path+'phecode_definitions1.2.csv')
ref_procedure=pd.read_csv(ref_path+'CPT_dictionary.csv')
ref_drug=pd.read_csv(ref_path+'RXCUI_dic_hao.csv')

phi_diag=np.load(model_path+'Phi_diag_mean.npy')
phi_drug=np.load(model_path+'Phi_drug_mean.npy')
phi_proc=np.load(model_path+'Phi_procedure_mean.npy')

df=pd.read_csv(save_path+'top'+str(top_N)+'_code_in_all_topics.tsv',sep='\t')

### read key topics
key_f=open(save_path+'key_topic.txt','r')
key=key_f.readlines()
topic_num_list=[int(i.rstrip()) for i in key]
key_f.close()

########
df=df.loc[df['topic'].isin(['topic'+str(i) for i in topic_num_list])]
diag_name_list=[]
diag_label_list=[]
drug_name_list=[]
proc_name_list=[]
for i in range(df.shape[0]):
    diag_id=df.iloc[i,:]['diagnosis']
    drug_id=df.iloc[i,:]['drug']
    proc_id=df.iloc[i,:]['procedure']

    diag_name=ref_diag.loc[ref_diag['phecode']==diag_id]['phenotype']
    diag_label=ref_diag.loc[ref_diag['phecode']==diag_id]['category']
    drug_name=ref_drug.loc[ref_drug['RXCUI']==drug_id]['NAME']
    proc_name=ref_procedure.loc[ref_procedure['CPT_Code']==proc_id]['CPT_Label']

    if diag_name.shape[0]>0:
        diag_name=diag_name.tolist()
        diag_name_list+=diag_name
        diag_label_list+=diag_label.tolist()
    else:
        diag_name_list+=[diag_id]
        diag_label_list += ['other']

    if drug_name.shape[0]>0:
        drug_name = drug_name.tolist()
        if drug_name != ['NF']:
            drug_name_list+=drug_name
        else:
            drug_name_list += [drug_id]
    else:
        drug_name_list+=[drug_id]

    if proc_name.shape[0]>0:
        proc_name=proc_name.tolist()
        proc_name_list+=proc_name
    else:
        proc_name_list+=[proc_id]

df['diagosis_name']=diag_name_list
df['diagosis_label']=diag_label_list
df['drug_name']=drug_name_list
df['procedure_name']=proc_name_list

#### get topic weight
diag_weight_list=[]
drug_weight_list=[]
proc_weight_list=[]
for topic_num in topic_num_list:
    one_topic=phi_diag[:,topic_num]
    ind=one_topic.argsort()[-top_N:][::-1]
    values=one_topic[ind].tolist()
    diag_weight_list+=values

    one_topic=phi_drug[:,topic_num]
    ind=one_topic.argsort()[-top_N:][::-1]
    values=one_topic[ind].tolist()
    drug_weight_list+=values

    one_topic=phi_proc[:,topic_num]
    ind=one_topic.argsort()[-top_N:][::-1]
    values=one_topic[ind].tolist()
    proc_weight_list+=values
    # for i in values:
    #     print(i)

df['diagosis_weight']=diag_weight_list
df['drug_weight']=drug_weight_list
df['procedure_weight']=proc_weight_list


###
df['topic']=[i.split('topic')[-1] for i in df['topic']]
df = df[['topic','diagosis_name','diagosis_weight','diagosis_label',
         'drug_name','drug_weight','procedure_name','procedure_weight']]
df.to_csv(save_path+'top'+str(top_N)+'_name_weight_in_key_topics.csv',index=False)



