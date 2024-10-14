import numpy as np
import pandas as pd
import collections
import pickle
import sys
import os

########### load data
model_path=sys.argv[1]#'./DMTM/python_code/trained_model_exclude_dementia/topic30/'
temp_data_path=sys.argv[2]#'./data_preparation/Temp_data_exclude_dementia/'#change to your path
mapping_path=sys.argv[3]#./data_preparation/dic/
save_path=sys.argv[4]#'./results/exclude_dementia/'
top_num=sys.argv[5]#10

if not os.path.exists(save_path):
    os.makedirs(save_path)

diag=np.load(model_path+'Phi_diag_mean.npy',allow_pickle=True)
diag_code2index = np.load(temp_data_path+'diag_code2index.npy', allow_pickle=True).tolist()
mappping_dic=pd.read_csv(mapping_path+'phecode_definitions1.2.csv',dtype='str')
top_num=int(top_num)

###########
key_topic=[]
key_topic_diag={}
for k in range(diag.shape[1]):
    ## diagnosis
    diag5 = diag[:, k]
    diag5_ind = diag5.argsort()[-top_num:][::-1]# top 5 index
    diag5_code = [list(diag_code2index.keys())[list(diag_code2index.values()).index(i)] for i in diag5_ind]
    diag5_categoty=[mappping_dic.loc[mappping_dic['phecode']==i]['category_number'].to_string(index=False) for i in diag5_code]
    counter = collections.Counter(diag5_categoty)
    if max(list(counter.values())) >= top_num/2:
        key_topic.append(k)
        # get diag code in the common category
        common_category_number=max(counter,key=counter.get)
        common_category=mappping_dic.loc[mappping_dic['category_number']==common_category_number]['category'].iloc[0]
        dict_name='T'+str(k)+'-'+common_category
        indices = [i for i, x in enumerate(diag5_categoty) if x == common_category_number]
        dict_value=list(np.array(diag5_code)[indices])
        key_topic_diag[dict_name]=dict_value
    # if len(set(diag5_categoty)) < (top_num-4):#at least 5 common category
    #     key_topic.append(k)

out=open(save_path+'key_topic.txt','w')
for i in key_topic:
    out.write(str(i)+'\n')

out.close()

with open(save_path+'key_topic_diag.pkl', 'wb') as f:
    pickle.dump(key_topic_diag, f)

