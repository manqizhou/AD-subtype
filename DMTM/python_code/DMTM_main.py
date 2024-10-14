"""
===========================================
DMTM
===========================================

"""

# Author: Hao
# License: Apache License Version 2.0


import numpy as np
import scipy.io as sio
from scipy.sparse import coo_matrix
from pydpm.model import PGDS_multi_sample
import sys
import os

data_path=sys.argv[1]#'../data/'
model_path=sys.argv[2]#'./trained_model/'
num_topic=sys.argv[3]#30

if not os.path.exists(model_path):
    os.makedirs(model_path)

##
train_data_diagnosis = np.load(data_path+'Data_diag.npy', allow_pickle=True).item().toarray()
train_data_diagnosis[train_data_diagnosis>1] = 1
train_data_diagnosis = np.array(train_data_diagnosis, order='C')
Patient_label_diagnosis = np.load(data_path+'Patient_label_diag.npy', allow_pickle=True)

train_data_drug = np.load(data_path+'Data_drug.npy', allow_pickle=True).item().toarray()
train_data_drug[train_data_drug>1] = 1
train_data_drug = np.array(train_data_drug, order='C')
Patient_label_drug = np.load(data_path+'Patient_label_drug.npy', allow_pickle=True)

train_data_procedure = np.load(data_path+'Data_proc.npy', allow_pickle=True).item().toarray()
train_data_procedure[train_data_procedure>1] = 1
train_data_procedure = np.array(train_data_procedure, order='C')
Patient_label_procedure = np.load(data_path+'Patient_label_proc.npy', allow_pickle=True)

if np.sum((Patient_label_drug-Patient_label_diagnosis)!=0)>0:
    print('Warning: diagnosis and drug have different labels')
if np.sum((Patient_label_procedure-Patient_label_diagnosis)!=0)>0:
    print('Warning: diagnosis and procedure have different labels')
Patient_label = Patient_label_diagnosis

ii_diag, jj_diag = np.nonzero(train_data_diagnosis)
iijj_diag = np.nonzero(train_data_diagnosis.flatten())[0]

ii_drug, jj_drug = np.nonzero(train_data_drug)
iijj_drug = np.nonzero(train_data_drug.flatten())[0]

ii_procedure, jj_procedure = np.nonzero(train_data_procedure)
iijj_procedure = np.nonzero(train_data_procedure.flatten())[0]

#K=30
K=int(num_topic)
Phi_diag = np.random.rand(train_data_diagnosis.shape[0], K)
Phi_diag = Phi_diag / np.sum(Phi_diag, axis=0)

Phi_procedure = np.random.rand(train_data_procedure.shape[0], K)
Phi_procedure = Phi_procedure / np.sum(Phi_procedure, axis=0)

Phi_drug = np.random.rand(train_data_drug.shape[0], K)
Phi_drug = Phi_drug / np.sum(Phi_drug, axis=0)

model=PGDS_multi_sample(K, 'cpu')
model.initial(train_data_diagnosis, train_data_drug, train_data_procedure,
              Patient_label, Phi_diag, Phi_drug, Phi_procedure,
              ii_diag, jj_diag, iijj_diag,
              ii_drug, jj_drug, iijj_drug,
              ii_procedure, jj_procedure, iijj_procedure)
model.train(iter_all=500,model_path=model_path)


