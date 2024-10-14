import numpy as np
import scipy.io as sio
from tqdm import tqdm
import pandas as pd
from collections import Counter
import time as tt
import os
import sys


num_cluster = sys.argv[1]#5
num_cluster=int(num_cluster)
save_path = sys.argv[2]#'./'+str(num_cluster)+'cluster_'+version+'/'
model_path = sys.argv[3]#'../DMTM/python_code/trained_model_'+version+'/'+optimal_topic+'/'
temp_data_path = sys.argv[4]#'../data_preparation/Temp_data_'+version+'/'
key_topic_path = sys.argv[5]#'../results/'+version+'/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

### read key topics
key_topic_f=open(key_topic_path+'key_topic.txt','r')
key_topic=key_topic_f.readlines()
key_topic=[int(i) for i in key_topic]
###
Theta = np.load(model_path+'Theta_mean.npy')
Theta=Theta[key_topic,:]
Theta_original = np.copy(Theta)
Theta = (Theta-np.min(Theta, axis=0)) / (np.max(Theta, axis=0)-np.min(Theta, axis=0))

Patient_label = np.load(temp_data_path+'Patient_label_proc.npy') # change
Time_sheet = np.load(temp_data_path+'time_sheet_condition_minLength_2_Age_Larger_than_50.npy') # change
Time_sheet = Time_sheet/12

demo = pd.read_csv(temp_data_path+'demo_condition_minLength_2_Age_Larger_than_50.csv') # change

if 'HMCI2AD_time' in demo.columns:
    mci2ad_time_varName='HMCI2AD_time'
elif 'MCI2AD_time' in demo.columns:
    mci2ad_time_varName = 'MCI2AD_time'
else:
    print('Error in the demo column name: no MCI2AD_time or HMCI2AD_time')
    sys.exit()

patient_select_flag = []
for i in range(demo.shape[0]):
    index = (Patient_label == i)
    if (demo.iloc[i,:][mci2ad_time_varName]>=365*1) and ((Time_sheet[index] <= 5).sum()>2): # > 2 vists; progressiont time > 1yr
        patient_select_flag.append(True)
    else:
        patient_select_flag.append(False)

real_patient_num = np.array(patient_select_flag).sum()

if Theta.size ==Theta.shape[0]:
    num_feature = 1
else:
    num_feature = Theta.shape[0]

num_patient = int(Patient_label.max()+1)

Select_data = []
Select_data_original = []
Select_timesheet = []
max_length = 0
max_year = 0
for i in range(num_patient):
    if patient_select_flag[i]:
        index = (Patient_label == i)
        if False in list((Time_sheet[index]<=5)):
            ttt = list((Time_sheet[index]<=5)).index(False)
            Select_data.append(Theta[:, index][:,0: ttt])
            Select_data_original.append(Theta_original[:, index][:,0: ttt])
            Select_timesheet.append(Time_sheet[index][0: ttt])
            max_year = np.maximum(max_year, (Time_sheet[index][0:ttt]).max())
            max_length = np.maximum(max_length, (Theta[:, index][:, 0:ttt]).shape[1])
        else:
            Select_data.append(Theta[:, index])
            Select_data_original.append(Theta_original[:, index])
            Select_timesheet.append(Time_sheet[index])
            max_year = np.maximum(max_year, Time_sheet[index].max())
            max_length = np.maximum(max_length, Theta[:, index].shape[1])


###########################
X = np.zeros([real_patient_num, max_length, num_feature])
Mask = np.zeros([real_patient_num, max_length, num_feature])
length_new = np.zeros(real_patient_num)
for i in range(real_patient_num):

    theta = np.transpose(Select_data[i])

    length = len(Select_timesheet[i])
    X[i, :length, :] = theta
    Mask[i, :length, :] = 1
    length_new[i] = length

#########################

Beta = 1e-1*np.random.randn(num_feature, 4, num_cluster)
Alpha = 1/num_cluster * np.ones(num_cluster)
sigma = 0.1*np.ones([num_feature])

Tau = np.zeros([real_patient_num, max_length, 4])
for i in range(real_patient_num):
    length = length_new[i]
    time = Select_timesheet[i]
    for t in range(int(length)):
       Tau[i, t, :] = np.array([1, time[t], time[t] ** 2, time[t] ** 3])
       #Tau[i, t, :] = np.array([1, t/(length-1), (t/(length-1)) ** 2, (t/(length-1)) ** 3])

num_iteration = 1000
c = 2*np.pi

Matrix_tmp = np.zeros([real_patient_num, num_cluster])

for i in range(num_iteration):
    start = tt.time()

    Tau_tmp = np.repeat(Tau[:, np.newaxis, :, :], num_feature, axis=1)
    Beta_tmp = np.repeat(Beta[np.newaxis, :, :], real_patient_num, axis=0)

    mu = np.matmul(Tau_tmp, Beta_tmp).transpose(0,2,3,1) # N L K D
    ## E step
    for k in range(num_cluster):
        tmp = ((X - mu[:, :, k, :]) / sigma) ** 2
        tmp2 = (-0.5*tmp)*Mask
        Matrix_tmp[:, k] = tmp2.sum(1).sum(1) + np.log(Alpha[k])
        #Matrix_tmp[:, k] = tmp2.sum(1).sum(1) + np.log(Alpha[k]) - length_new*np.log(sigma[k])

    Matrix_tmp = Matrix_tmp - Matrix_tmp.max(1, keepdims=True)

    Probability = np.exp(Matrix_tmp) / np.exp(Matrix_tmp).sum(1, keepdims=True)

    ## M step
    # Update alpha
    Alpha = Probability.sum(0)/real_patient_num

    # Update beta
    for k in range(num_cluster):
        for j in range(4):
            for d in range(num_feature):
                Tau_tmp = np.repeat(Tau[:, np.newaxis, :, :], num_feature, axis=1)
                Beta_tmp = np.repeat(Beta[np.newaxis, :, :], real_patient_num, axis=0)
                mu = np.matmul(Tau_tmp, Beta_tmp).transpose(0,2,3,1)
                tmp = Tau[:, :, j]*Beta[d, j, k]
                A = X[:,:,d] - mu[:,:,k,d]+tmp
                A = A * Mask[:,:,0]
                if ((Probability[:, k][:, np.newaxis]*((Tau[:, :, j])**2)).sum()) > 0:
                    Beta[d,j,k] = ((Probability[:, k][:, np.newaxis]*A*Tau[:, :, j]).sum()) / ((Probability[:, k][:, np.newaxis]*((Tau[:, :, j])**2)).sum())

    # Update sigma
    fenzi = 0
    fenmu = 0

    Tau_tmp = np.repeat(Tau[:, np.newaxis, :, :], num_feature, axis=1)
    Beta_tmp = np.repeat(Beta[np.newaxis, :, :], real_patient_num, axis=0)
    mu = np.matmul(Tau_tmp, Beta_tmp).transpose(0, 2, 3, 1)  # N L K D

    for d in range(num_feature):
        for k in range(num_cluster):
            fenzi += (Probability[:, k][:, np.newaxis] * ((X[:,:,d] - mu[:, :, k, d]) ** 2)).sum()
            fenmu += (Probability[:, k] * length_new).sum()

        sigma[d] = np.sqrt(fenzi/fenmu)

    ## Calculate loss function
    Tau_tmp = np.repeat(Tau[:, np.newaxis, :, :], num_feature, axis=1)
    Beta_tmp = np.repeat(Beta[np.newaxis, :, :], real_patient_num, axis=0)
    mu = np.matmul(Tau_tmp, Beta_tmp).transpose(0, 2, 3, 1)  # N L K D
    LL = 0
    for k in range(num_cluster):
        tmp1 = ((X - mu[:, :, k, :]) / sigma) ** 2
        #tmp2 = np.prod(sigma)**(length_new)
        #tmp3 = (np.sqrt(c)) **(length_new*num_feature)

        #tmp4 = np.exp((length_new*num_feature/2) * np.log(c) + length_new*np.log(np.prod(sigma)))

        tmp2 = 1/(np.sqrt(c) * sigma[np.newaxis, np.newaxis, :]) * np.exp((-0.5 * tmp1))
        tmp2[Mask == 0] = 1

        #LL = LL + Alpha[k] * np.exp((-0.5 * tmp1).sum(1).sum(1)) * 1 / (tmp4)
        LL = LL + Alpha[k] * tmp2.prod(1).prod(1)

    LL[LL == 0.] = 1e-100
    LL[LL == np.inf] = 1e100
    #LL[np.isnan(LL)] = 1e-100
    likelihood = np.sum(np.log(LL))

    end = tt.time()
    if i%100==0:
        print('Iteration: %d Likelihood: %.4f Time: %.4f Max_alpha: %.4f Min_alpha: %.4f num_class: %.2f' %
              (i, likelihood, end - start, Alpha.max(), Alpha.min(), len(np.unique(Probability.argmax(1)))))
        print(Counter(Probability.argmax(1)))

    save_obj=[Beta, Alpha, Probability, sigma, Select_data_original, Select_data, Select_timesheet]
    arr = np.asanyarray(save_obj,dtype=object)
    np.save(save_path+'T_GBTM_result_all.npy', arr) # change
    


