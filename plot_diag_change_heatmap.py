import numpy as np
import pandas as pd
from collections import Counter
import math
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

temp_data_path=sys.argv[1]#'./data_preparation/'+temp_folder+'/'
key_topic_path=sys.argv[2]#'./results/'+version+'/'
KM_result_path = sys.argv[3]#'./results/'+version + '/' + cluster_name + '/KM/'
save_path = sys.argv[4]#'./results/'+version + '/' + cluster_name+'/cluster_topic_change/'

plot_save_path=save_path+'heatmap/'

if not os.path.exists(plot_save_path+'useFirstDiag'):
    os.makedirs(plot_save_path+'useFirstDiag')

if not os.path.exists(plot_save_path+'useAllDiag'):
    os.makedirs(plot_save_path+'useAllDiag')


cdf_count_useAll=np.load(save_path+'useAllDiag_cluster_topic_change_histogram.npy')
cdf_count_useFirst=np.load(save_path+'useFirstDiag_cluster_topic_change_histogram.npy')

km_df=pd.read_csv(KM_result_path+'MCI2AD_time_cluster_df.csv')
key_topics=open(key_topic_path+'key_topic.txt','r').readlines()#[5,10,15,16,18,19,22,25,27,28]
key_topics=[int(i.rstrip()) for i in key_topics]
num_topic=len(key_topics)


cluster_size=Counter(km_df.Cluster)
num_cluster=np.max(km_df.Cluster)


## get time
Patient_label=np.load(temp_data_path+'Patient_label_diag.npy',allow_pickle=True)
time=np.load(temp_data_path+'time_sheet_condition_minLength_2_Age_Larger_than_50.npy')
Time_sheet = time/12
demo = pd.read_csv(temp_data_path+'demo_condition_minLength_2_Age_Larger_than_50.csv')

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
Patient_label=Patient_label[patient_keep_index]
time=time[patient_keep_index]
time_point=list(set(time))


time_window=12#12 month, 1yr
num_window=5#2,3,4,5,>5


cluster_size_per_window_dict={}
for c in range(num_cluster):
    one_cluster=km_df.loc[km_df.Cluster==(c+1)]
    cluster_size_per_window_list=[]
    for w in range(num_window):
        min_t=(w+1)*365#start from 2nd yer
        if w==num_window-1:#the last window
            max_t=np.max(one_cluster.MCI2AD_time)+1
        else:
            max_t=(w+2)*365
        n=one_cluster.loc[(one_cluster.MCI2AD_time>=min_t) & (one_cluster.MCI2AD_time<max_t)]
        cluster_size_per_window_list.append(n.shape[0])
    cluster_size_per_window_dict[c]=cluster_size_per_window_list
    

time_window_ind=[]
for w in range(num_window):
    min_t=(w+1)*12#in month
    if w==num_window-1:#the last window
        max_t=np.max(one_cluster.MCI2AD_time)/12+1
    else:
        max_t=(w+2)*12#in month
    indices = [i for i, x in enumerate(time_point) if (x>=min_t) & (x<max_t)]
    time_window_ind.append(indices)


cluster_new_count_useAll = np.empty((0,num_window), float)
cluster_new_count_useFirst = np.empty((0,num_window), float)
topic_diag_count_useAll=[0]*num_topic
topic_diag_count_useFirst=[0]*num_topic
for c in range(num_cluster):
    one_cluster_count_useAll=cdf_count_useAll[(c*num_topic):(c*num_topic+num_topic),:]# use all
    one_cluster_count_useFirst=cdf_count_useFirst[(c*num_topic):(c*num_topic+num_topic),:]# use first
    
    one_cluster_size=cluster_size[c+1]
    for t in range(num_topic):
        one_topic_new_count_useAll=np.array([0.0]*num_window)
        one_topic_new_count_useFirst=np.array([0.0]*num_window)
        one_cluster_one_topic_count_useAll=one_cluster_count_useAll[t,:]*one_cluster_size
        one_cluster_one_topic_count_useFirst=one_cluster_count_useFirst[t,:]*one_cluster_size
        
        topic_diag_count_useAll[t]+=one_cluster_one_topic_count_useAll.sum()
        topic_diag_count_useFirst[t]+=one_cluster_one_topic_count_useFirst.sum()

        for i in range(num_window):
            ind=time_window_ind[i]
            sub_useAll=one_cluster_one_topic_count_useAll[ind].sum()
            sub_useFirst=one_cluster_one_topic_count_useFirst[ind].sum()
            if i==0:
                non_AD_num=one_cluster_size
            else:
                non_AD_num=one_cluster_size-sum(cluster_size_per_window_dict[c][:i])          
            one_topic_new_count_useAll[i]=sub_useAll/non_AD_num
            one_topic_new_count_useFirst[i]=sub_useFirst/non_AD_num
        
        one_topic_new_count_useAll=one_topic_new_count_useAll.reshape(1,num_window)
        cluster_new_count_useAll = np.append(cluster_new_count_useAll, one_topic_new_count_useAll, axis=0)
        
        one_topic_new_count_useFirst=one_topic_new_count_useFirst.reshape(1,num_window)
        cluster_new_count_useFirst = np.append(cluster_new_count_useFirst, one_topic_new_count_useFirst, axis=0)
    
topic_diag_count_useAll=[i/km_df.shape[0] for i in topic_diag_count_useAll]#avg count for the topic codes for each person
topic_diag_count_useFirst=[i/km_df.shape[0] for i in topic_diag_count_useFirst]
for t in range(num_topic):
    cluster_new_count_useAll[[t,t+num_topic,t+num_topic*2,t+num_topic*3,t+num_topic*4],:]=cluster_new_count_useAll[[t,t+num_topic,t+num_topic*2,t+num_topic*3,t+num_topic*4],:]/topic_diag_count_useAll[t]
    cluster_new_count_useFirst[[t,t+num_topic,t+num_topic*2,t+num_topic*3,t+num_topic*4],:]=cluster_new_count_useFirst[[t,t+num_topic,t+num_topic*2,t+num_topic*3,t+num_topic*4],:]/topic_diag_count_useFirst[t]
    

np.save(save_path+'useAllDiag_cluster_topic_change_heatmap.npy', cluster_new_count_useAll)
np.save(save_path+'useFirstDiag_cluster_topic_change_heatmap.npy', cluster_new_count_useFirst)
np.savetxt(save_path+"useAllDiag_heatmap.csv", cluster_new_count_useAll, delimiter=",")
np.savetxt(save_path+"useFirstDiag_heatmap.csv", cluster_new_count_useFirst, delimiter=",")


for c in range(num_topic):
    sub=cluster_new_count_useAll[[c,c+num_topic,c+num_topic*2,c+num_topic*3,c+num_topic*4],:]
    sub=pd.DataFrame(sub,columns=['2yr','3yr','4yr','5yr','>5yr'],index=['C1','C2','C3','C4','C5'])
    ax=sns.heatmap(sub, annot=False, linewidth=.5,cmap="Blues")
    ax.set_title('T'+str(key_topics[c]))
    plt.savefig(plot_save_path+"useAllDiag/heatmap_T"+str(c)+".png")
    plt.clf()
    
    sub=cluster_new_count_useFirst[[c,c+num_topic,c+num_topic*2,c+num_topic*3,c+num_topic*4],:]
    sub=pd.DataFrame(sub,columns=['2yr','3yr','4yr','5yr','>5yr'],index=['C1','C2','C3','C4','C5'])
    ax=sns.heatmap(sub, annot=False, linewidth=.5,cmap="Blues")
    ax.set_title('T'+str(key_topics[c]))
    plt.savefig(plot_save_path+"useFirstDiag/heatmap_T"+str(c)+".png")
    plt.clf()


for c in range(num_cluster):
    sub=cluster_new_count_useAll[c*num_topic:(c+1)*num_topic,:]
    sub=pd.DataFrame(sub,columns=['2yr','3yr','4yr','5yr','>5yr'],index=['T'+str(key_topics[i]) for i in range(num_topic)])
    ax=sns.heatmap(sub, annot=False, linewidth=.5,cmap="Blues")
    ax.set_title('C'+str(c))
    plt.savefig(plot_save_path+"useAllDiag/heatmap_C"+str(c)+".png")
    plt.clf()
    
    sub=cluster_new_count_useFirst[c*num_topic:(c+1)*num_topic,:]
    sub=pd.DataFrame(sub,columns=['2yr','3yr','4yr','5yr','>5yr'],index=['T'+str(key_topics[i]) for i in range(num_topic)])
    ax=sns.heatmap(sub, annot=False, linewidth=.5,cmap="Blues")
    ax.set_title('C'+str(c))
    plt.savefig(plot_save_path+"useFirstDiag/heatmap_C"+str(c)+".png")
    plt.clf()

