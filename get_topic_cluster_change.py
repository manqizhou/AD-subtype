####### 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
import os

##### load data
cluster_path=sys.argv[1]#'./TCLA/5cluster_dementia_excludePT1_keyTopic_rep2/'
model_path=sys.argv[2]#'./DMTM/python_code/trained_model_exclude_dementia/topic30/'
key_topic_path=sys.argv[3]#'./results/exclude_dementia/key_topic.txt'
save_path = sys.argv[4]#'./results/exclude_dementia/5cluster_keyTopic/rep2/'### change

plot_save_path = save_path+'python_plot/'

if not os.path.exists(plot_save_path):
    os.makedirs(plot_save_path)

######
Data = np.load(cluster_path+'T_GBTM_result_all.npy', allow_pickle=True)
phi_diag=np.load(model_path+'Phi_diag_mean.npy')
phi_drug=np.load(model_path+'Phi_drug_mean.npy')
phi_proc=np.load(model_path+'Phi_procedure_mean.npy')
Theta=np.load(model_path+'Theta_mean.npy')

##
key_topic_f=open(key_topic_path+'key_topic.txt','r')
key_topic=key_topic_f.readlines()
key_topic=[int(i.rstrip()) for i in key_topic]
phi_diag=phi_diag[:,key_topic]

#########
Beta = Data[0]
Select_timesheet = Data[6]
Probability = Data[2]

################
num_key_topic = len(key_topic)
Time_line_original = np.arange(0, 60 + 1, 3)
Time_line = Time_line_original/12
Tau = np.zeros((len(Time_line),4))
for t in range(len(Time_line)):
    Tau[t, :] = np.array([1, Time_line[t], Time_line[t] ** 2, Time_line[t] ** 3])

####
num_cluster=Beta.shape[2]
#num_cluster=len(combined_clusters)
num_time=Tau.shape[0]

Mu=np.zeros([num_cluster, num_key_topic, num_time])
for cluster_num in range(num_cluster):
    beta1 = Beta[:, :,cluster_num]
    #beta1=Beta[:,:,combined_clusters[cluster_num]]
    #beta1=np.sum(beta1,axis=2)
    mu=np.matmul(beta1,Tau.transpose())#beta * tau
    for j in range(num_key_topic):
        Mu[cluster_num,j,:]=mu[j]

## save in npy
np.save(save_path+'Mu.npy',Mu)
## save in csv
mu_flat=Mu.reshape(num_cluster*num_key_topic,num_time)
cluster_index=[i for i in range(num_cluster) for j in range(num_key_topic)]
cluster_index=np.array(cluster_index).reshape(len(cluster_index),1)
mu_flat=np.append(cluster_index,mu_flat,axis=1)
np.savetxt(save_path+'Mu.csv', mu_flat, delimiter=',')


#### plot
colors=list(mcolors.TABLEAU_COLORS.values())
LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']

#topic change in each cluster
Mu[Mu<0]=0
mu_mean=np.mean(Mu)
ymin=np.min(Mu)
ymax=max(np.max(Mu),1)
for i in range(num_cluster):
    fig, ax = plt.subplots(1, 1, figsize=(16, 9), dpi=120)
    for j in range(num_key_topic):
        if np.min(Mu[i,j,:])<mu_mean:
            alpha=0.1
        else:
            alpha=1
        plt.plot(Time_line_original, Mu[i,j,:],label=key_topic[j],color=colors[j%10],linestyle=LINE_STYLES[j//10],alpha=alpha)
    plt.axhline(y=mu_mean, color='r', linestyle='-')
    plt.ylim(ymin, ymax)
    plt.legend()
    #plt.show()
    plt.savefig(plot_save_path+'Mu_cluster' + str(i) + '.png')

#cluster change in each topic
for i in range(num_key_topic):
    fig, ax = plt.subplots(1, 1, figsize=(16, 9), dpi=120)
    for j in range(num_cluster):
        plt.plot(Time_line_original, Mu[j,i,:],label=j,color=colors[j%10],linestyle=LINE_STYLES[j//10])
    plt.ylim(ymin,ymax)
    plt.legend()
    #plt.show()
    plt.savefig(plot_save_path+'Mu_topic' + str(key_topic[i]) + '.png')

##
#print('done')
