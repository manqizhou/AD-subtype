import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import sys
import os

######
model_path=sys.argv[1]
key_topic_path=sys.argv[2]
save_path=sys.argv[3]

if not os.path.exists(save_path):
    os.makedirs(save_path)
####
pi_mean=np.load(model_path+'Pi_mean.npy')
### read key topics
key_topic_f=open(key_topic_path+'key_topic.txt','r')
key_topic=key_topic_f.readlines()
key_topic=[int(i) for i in key_topic]

num_key=len(key_topic)
num_topic=pi_mean.shape[0]

others=[i for i in list(range(0,num_topic)) if i not in key_topic]#changed 20 to num_topic
pi_key=pi_mean[key_topic,:][:,key_topic]
pi_key_others=np.mean(pi_mean[key_topic,:][:,others],axis=1).reshape((num_key,1))#changed this line
pi_others_key=np.sum(pi_mean[others,:][:,key_topic],axis=0).reshape((1,num_key))#changed this line
pi_others_others=(1-np.sum(pi_key_others)).reshape((1,1))
pi_transformation=np.concatenate((np.append(pi_key,pi_key_others,axis=1),np.append(pi_others_key,pi_others_others,axis=1)))

pi_plot_df=pd.DataFrame(data=pi_transformation,index=['T'+str(i) for i in key_topic]+['others'],columns=['T'+str(i) for i in key_topic]+['others'])
pi_plot_df.to_csv(save_path+'transition.csv')

##### plot heatmap
plt.figure(figsize=(10, 8))
hm = sn.heatmap(data=pi_plot_df*100, annot=True,cmap="Blues")

plt.savefig(save_path+'transition.png',dpi=300)