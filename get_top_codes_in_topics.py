############################## This code is used to get supp fig 11 plot data##############################
###### this code can get top10 codes in all topics

import numpy as np
import sys
import os

### load data
model_path = sys.argv[1]#'./DMTM/python_code/trained_model_exclude_dementia/topic30/'#change to your path
temp_data_path = sys.argv[2]#'./data_preparation/Temp_data_exclude_dementia/'#change to your path
save_path = sys.argv[3]#'./results/exclude_dementia/'#change to your path
top_num = int(sys.argv[4])#10

diag=np.load(model_path+'Phi_diag_mean.npy')
drug=np.load(model_path+'Phi_drug_mean.npy')
procedure=np.load(model_path+'Phi_procedure_mean.npy')
Theta=np.load(model_path+'Theta_mean.npy')

diag_code2index = np.load(temp_data_path+'diag_code2index.npy', allow_pickle=True).tolist()
drug_code2index = np.load(temp_data_path+'drug_code2index.npy', allow_pickle=True).tolist()
proc_code2index = np.load(temp_data_path+'procedure_code2index.npy', allow_pickle=True).tolist()

if not os.path.exists(save_path):
    os.makedirs(save_path)

#####
out=open(save_path+'top'+str(top_num)+'_code_in_all_topics.tsv','w')
out.write('topic\tdiagnosis\tdrug\tprocedure\n')

## top features in all topics
#for k in key_topic:
for k in range(diag.shape[1]):
    ## diagnosis
    diag5 = diag[:, k]
    diag5_ind = diag5.argsort()[-top_num:][::-1]# top 5 index
    diag5_code = [list(diag_code2index.keys())[list(diag_code2index.values()).index(i)] for i in diag5_ind]

    ## drug
    drug5 = drug[:, k]
    drug5_ind = drug5.argsort()[-top_num:][::-1]
    drug5_code = [list(drug_code2index.keys())[list(drug_code2index.values()).index(i)] for i in drug5_ind]

    ## procedure
    proc5 = procedure[:, k]
    proc5_ind = proc5.argsort()[-top_num:][::-1]
    proc5_code = [list(proc_code2index.keys())[list(proc_code2index.values()).index(i)] for i in proc5_ind]
    
    for i in range(top_num):
        out.write('topic%i\t%s\t%s\t%s\n' % (k,diag5_code[i],drug5_code[i],proc5_code[i]))

out.close()

