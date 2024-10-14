import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import pickle
import os
from collections import Counter
import sys

temp_data_path=sys.argv[1]#'./data_preparation/Temp_data_rm_all_AD_MCI_dementia_related_diag/'
key_topic_path=sys.argv[2]#'./results/exclude_related_diag/'
cluster_path=sys.argv[3]#'./TCLA/5cluster_relatedDiag_excludePT1_keyTopic/'
save_path=sys.argv[4]#'./results/exclude_related_diag/topic_cluster_change_useAlltime/'

plot_save_path=save_path+'histogram_cdf/'


if not os.path.exists(plot_save_path+'useFirstDiag'):
    os.makedirs(plot_save_path+'useFirstDiag')

if not os.path.exists(plot_save_path+'useAllDiag'):
    os.makedirs(plot_save_path+'useAllDiag')

#### read files
data_diag=np.load(temp_data_path+'Data_diag.npy',allow_pickle=True).item().toarray()
diag_code2index=np.load(temp_data_path+'diag_code2index.npy',allow_pickle=True).tolist()
data_diag[data_diag>1] = 1

Patient_label=np.load(temp_data_path+'Patient_label_diag.npy',allow_pickle=True)
time=np.load(temp_data_path+'time_sheet_condition_minLength_2_Age_Larger_than_50.npy')
Time_sheet = time/12

demo = pd.read_csv(temp_data_path+'demo_condition_minLength_2_Age_Larger_than_50.csv')
Data=np.load(cluster_path+'T_GBTM_result_all.npy', allow_pickle=True)
Probability = Data[2]#probability
num_cluster=Probability.shape[1]

## read key topics
key_topic_f=open(key_topic_path+'key_topic.txt','r')
key_topic_list=key_topic_f.readlines()
key_topic_list=[int(i.rstrip()) for i in key_topic_list]
num_key_topic=len(key_topic_list)
## read phecodes in key topics that we want to plot
with open(key_topic_path+'key_topic_diag.pkl', 'rb') as f:
    key_topic_diag = pickle.load(f)

##
if 'HMCI2AD_time' in demo.columns:
    mci2ad_time_varName='HMCI2AD_time'
elif 'MCI2AD_time' in demo.columns:
    mci2ad_time_varName = 'MCI2AD_time'
else:
    print('Error in the demo column name: no MCI2AD_time or HMCI2AD_time')
    sys.exit()

patient_keep_index = []
for i in range(demo.shape[0]):
    index = (Patient_label == i)
    if (demo.iloc[i,:][mci2ad_time_varName]>=365*1) and ((Time_sheet[index] <= 5).sum()>2): # change column to MCI2AD_time
        indices = [p for p, x in enumerate(Patient_label) if x == i]
        patient_keep_index+=indices
    else:
        pass


## data for patients used in clustering
data_diag=data_diag[:,patient_keep_index]
Patient_label=Patient_label[patient_keep_index]
time=time[patient_keep_index]
time_point=list(set(time))


def plot_topic_count_per_cluster(cluster_number,cluster_size, diag_name,time_point,diag_list,
                                 data_diag_one_cluster,Patient_label_one_cluster,time_one_cluster):
    ## get the index of interested diag
    diag_index_list=[diag_code2index[i] for i in diag_list]
    ## get the rows corresponding with the interested diag
    diag_sub=data_diag_one_cluster[diag_index_list,]
    diag_sub=diag_sub.sum(axis=0)#column sum
    diag_sub[diag_sub>1]=1
    
    ## get count
    diag_count_useFirst = np.empty((0,len(time_point)), int)# (patient x time) diag count
    diag_count_useAll = np.empty((0, len(time_point)), int)# (patient x time) diag count

    for p in set(Patient_label_one_cluster):# for each patient
        count_useFirst=np.array([0]*len(time_point))
        count_useAll = np.array([0] * len(time_point))

        patient_index=Patient_label_one_cluster==p
        patient_time=time_one_cluster[patient_index]
        patient_diag=diag_sub[patient_index]

        if np.max(patient_diag)==0:#no diag
            count_useFirst=count_useFirst
            count_useAll=count_useAll
        else:
            # only use the first diag
            first_diag_time=patient_time[patient_diag==1][0]
            ind=time_point.index(first_diag_time)
            count_useFirst[ind]=1

            # use all diag
            all_diag_time=patient_time[patient_diag==1]
            ind=[time_point.index(i) for i in all_diag_time]
            count_useAll[ind]=1

        count_useFirst=count_useFirst.reshape(1,len(time_point))
        count_useAll = count_useAll.reshape(1, len(time_point))

        diag_count_useFirst = np.append(diag_count_useFirst, count_useFirst, axis=0)
        diag_count_useAll = np.append(diag_count_useAll, count_useAll, axis=0)
    
    ### plot histogram
    # use first diag
    prop=diag_count_useFirst.sum(axis=0)/cluster_size
    plt.hist(time_point, weights=prop, bins=len(time_point), density=False,color='#0504aa',rwidth=0.8,cumulative=False)
    plt.twinx()
    plt.plot(time_point , prop.cumsum(), c='red',label='CDF')
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Time')
    plt.ylabel('Patient Count')
    plt.title('C'+str(cluster_number+1)+'_'+diag_name)
    plt.legend(loc='lower right')
    save_name='useFirstDiag/C'+str(cluster_number+1)+'_'+diag_name.split('-')[0]+'.png'
    plt.savefig(plot_save_path+save_name)
    plt.clf()

    # use all diag
    prop = diag_count_useAll.sum(axis=0) / cluster_size
    plt.hist(time_point, weights=prop, bins=len(time_point), density=False, color='#0504aa',
             rwidth=0.8, cumulative=False)
    plt.twinx()
    plt.plot(time_point, prop.cumsum(), c='red', label='CDF')
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Time')
    plt.ylabel('Patient Count')
    plt.title('C' + str(cluster_number + 1) + '_' + diag_name)
    plt.legend(loc='lower right')
    save_name = 'useAllDiag/C' + str(cluster_number + 1) + '_' + diag_name.split('-')[0] + '.png'
    plt.savefig(plot_save_path + save_name)
    plt.clf()

    return diag_count_useFirst.sum(axis=0)/ cluster_size, diag_count_useAll.sum(axis=0)/ cluster_size


cluster_assign=np.argmax(Probability, axis=1)# cluster assignment for each patient
topic_change_all_cluster_useFirst=np.empty((0,len(time_point)), int)
topic_change_all_cluster_useAll=np.empty((0,len(time_point)), int)
for c in range(num_cluster):
    cluster_size=sum(cluster_assign==c)
    patient_indices = [p for p, x in enumerate(cluster_assign) if x == c]# patient index in cluster c
    indices = [i for i, value in enumerate(Patient_label) if value in patient_indices]
    data_diag_one_cluster=data_diag[:,indices]
    Patient_label_one_cluster=Patient_label[indices]
    time_one_cluster=time[indices]
    for diag_name in key_topic_diag.keys():
        diag_list=key_topic_diag[diag_name]
        topic_change_useFirst, topic_change_useAll = plot_topic_count_per_cluster(c, cluster_size, diag_name,time_point,
                                                                                  diag_list,data_diag_one_cluster,
                                                                                  Patient_label_one_cluster,time_one_cluster)

        topic_change_all_cluster_useFirst=np.append(topic_change_all_cluster_useFirst,topic_change_useFirst.reshape(1,len(time_point)), axis=0)
        topic_change_all_cluster_useAll = np.append(topic_change_all_cluster_useAll,
                                                      topic_change_useAll.reshape(1, len(time_point)), axis=0)

np.save(save_path+'useFirstDiag_cluster_topic_change_histogram.npy', topic_change_all_cluster_useFirst)
np.save(save_path+'useAllDiag_cluster_topic_change_histogram.npy', topic_change_all_cluster_useAll)
np.savetxt(save_path+"useAllDiag_histogram.csv", topic_change_all_cluster_useAll, delimiter=",")
np.savetxt(save_path+"useFirstDiag_histogram.csv", topic_change_all_cluster_useFirst, delimiter=",")

#### plot overall
topic_change_all_cluster=topic_change_all_cluster_useFirst
for c in range(num_cluster):
    topic_change_one_cluster=topic_change_all_cluster[(c*num_key_topic):((c+1)*num_key_topic),:]
    for t in range(num_key_topic):
        plt.plot(time_point, topic_change_one_cluster[t,:].cumsum(),label=list(key_topic_diag.keys())[t])
    plt.title('C'+str(c+1))
    plt.legend()
    plt.savefig(plot_save_path+'useFirstDiag/C'+str(c+1)+'.png')
    plt.clf()
for t in range(num_key_topic):
    topic_change_one_topic=topic_change_all_cluster[range(t,topic_change_all_cluster.shape[0],num_key_topic),:]
    diag_name=list(key_topic_diag.keys())[t]
    for c in range(num_cluster):
        plt.plot(time_point, topic_change_one_topic[c,:].cumsum(),label='C'+str(c+1))
    plt.title(diag_name)
    plt.legend()
    plt.savefig(plot_save_path+'useFirstDiag/'+diag_name.split('-')[0]+'.png')
    plt.clf()

topic_change_all_cluster=topic_change_all_cluster_useAll
for c in range(num_cluster):
    topic_change_one_cluster=topic_change_all_cluster[(c*num_key_topic):((c+1)*num_key_topic),:]
    for t in range(num_key_topic):
        plt.plot(time_point, topic_change_one_cluster[t,:].cumsum(),label=list(key_topic_diag.keys())[t])
    plt.title('C'+str(c+1))
    plt.legend()
    plt.savefig(plot_save_path+'useAllDiag/C'+str(c+1)+'.png')
    plt.clf()
for t in range(num_key_topic):
    topic_change_one_topic=topic_change_all_cluster[range(t,topic_change_all_cluster.shape[0],num_key_topic),:]
    diag_name=list(key_topic_diag.keys())[t]
    for c in range(num_cluster):
        plt.plot(time_point, topic_change_one_topic[c,:].cumsum(),label='C'+str(c+1))
    plt.title(diag_name)
    plt.legend()
    plt.savefig(plot_save_path+'useAllDiag/'+diag_name.split('-')[0]+'.png')
    plt.clf()

