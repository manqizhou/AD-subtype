############################## This code is used to get supp fig 14 plot data ##############################
###### supp fig 14 lower - stratified analysis
##### input: T_GBTM_result.npy (clustering result)
####         Patient_label_diag.npy
#####        condition_minLength_2_Age_Larger_than_50.csv (from preprocessing)
####         time_sheet_condition_minLength_2_Age_Larger_than_50.npy (from preprocessing)
##### output: plot data (age_class_gender_race.csv, PT_class_gender_race.csv)
#####         plot figure (1-10.png)
########################################################################################################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import sys
import os

def compute_p_value(data_df, group ='Gender', value = 'MCI_age_cluster'):## this function is based on 5 clusters
    if group=='Gender':
        df_M = list(data_df.loc[data_df['Gender']=='M',value])
        df_F = list(data_df.loc[data_df['Gender']=='F',value])
        min_len=min(len(df_M),len(df_F))
        if min_len<3:# cannot do shapiro test
            p_value = scipy.stats.kruskal(df_M, df_F)[1]
        else:
            if scipy.stats.shapiro(df_M)[1]>=0.05 and scipy.stats.shapiro(df_F)[1]>=0.05: # norm distribution
                p_value = scipy.stats.f_oneway(df_M,df_F)[1]
            else: # not norm distribution
                p_value = scipy.stats.kruskal(df_M,df_F)[1]

    if group =='Race':
        df_W = list(data_df.loc[data_df['Race'] == 'white', value])
        df_B = list(data_df.loc[data_df['Race'] == 'black', value])
        df_O = list(data_df.loc[data_df['Race'] == 'others', value])
        min_len = min(len(df_W), len(df_B),len(df_O))
        if min_len < 3:
            p_value = scipy.stats.kruskal(df_W, df_B, df_O)[1]
        else:
            if scipy.stats.shapiro(df_W)[1]>=0.05 and scipy.stats.shapiro(df_B)[1]>=0.05 and scipy.stats.shapiro(df_O)[1]>=0.05: # norm distribution
                p_value = scipy.stats.f_oneway(df_W,df_B,df_O)[1]
            else: # not norm distribution
                p_value = scipy.stats.kruskal(df_W,df_B,df_O)[1]

    if group=='Class':
        df_C1 = list(data_df.loc[data_df['Class'] == 1, value])
        df_C2 = list(data_df.loc[data_df['Class'] == 2, value])
        df_C3 = list(data_df.loc[data_df['Class'] == 3, value])
        df_C4 = list(data_df.loc[data_df['Class'] == 4, value])
        df_C5 = list(data_df.loc[data_df['Class'] == 5, value])
        min_len = min(len(df_C1), len(df_C2), len(df_C3), len(df_C4), len(df_C5))
        if min_len < 3:
            p_value = scipy.stats.kruskal(df_C1, df_C2, df_C3, df_C4, df_C5)[1]
        else:
            if scipy.stats.shapiro(df_C1)[1]>=0.05 and scipy.stats.shapiro(df_C2)[1]>=0.05 and scipy.stats.shapiro(df_C3)[1]>=0.05 and scipy.stats.shapiro(df_C4)[1]>=0.05 and scipy.stats.shapiro(df_C5)[1]>=0.05: # norm distribution
                p_value = scipy.stats.f_oneway(df_C1,df_C2,df_C3,df_C4,df_C5)[1]
            else: # not norm distribution
                p_value = scipy.stats.kruskal(df_C1,df_C2,df_C3,df_C4,df_C5)[1]

    return p_value

###### load data
cluster_path=sys.argv[1]
temp_data_path=sys.argv[2]
save_path=sys.argv[3]#'/Users/xuzhenxing/PycharmProjects/AD_subphenotype/oneFlorida_results_zhx/suppFig14/'

plot_save_path=save_path+'plot/'
if not os.path.exists(plot_save_path):
    os.makedirs(plot_save_path)


#####
Data = np.load(f'{cluster_path}T_GBTM_result_all.npy', allow_pickle=True)
Probability = Data[2]

demo = pd.read_csv(f'{temp_data_path}demo_condition_minLength_2_Age_Larger_than_50.csv')
Patient_label = np.load(f'{temp_data_path}Patient_label_diag.npy')
Time_sheet = np.load(f'{temp_data_path}time_sheet_condition_minLength_2_Age_Larger_than_50.npy')
Time_sheet = Time_sheet/12

#################
Sex = []
MCI_age = []
Region = []#race
MCI2AD_time = []

#### decide variable name
if 'HMCI2AD_time' in demo.columns:
    mci2ad_time_varName='HMCI2AD_time'
elif 'MCI2AD_time' in demo.columns:
    mci2ad_time_varName = 'MCI2AD_time'
else:
    print('Error in the demo column name: no MCI2AD_time or HMCI2AD_time')
    sys.exit()
#
if 'DSEX' in demo.columns:
    sex_varName='DSEX'
elif 'SEX' in demo.columns:
    sex_varName = 'SEX'
else:
    print('Error in the demo column name: no DSEX or SEX')
    sys.exit()
#
if 'GMCIAGE' in demo.columns:
    age_varName='GMCIAGE'
elif 'MCIAGE' in demo.columns:
    age_varName = 'MCIAGE'
else:
    print('Error in the demo column name: no GMCIAGE or MCIAGE')
    sys.exit()
#
if 'FRACE' in demo.columns:
    race_varName='FRACE'
elif 'RACE' in demo.columns:
    race_varName = 'RACE'
else:
    print('Error in the demo column name: no FRACE or RACE')
    sys.exit()


for i in range(demo.shape[0]):
    index = (Patient_label == i)
    if (demo.iloc[i, :][mci2ad_time_varName] >= 365 * 1) and ((Time_sheet[index] <= 5).sum() > 2):
        Sex.append(demo.iloc[i,:][sex_varName])
        MCI_age.append(demo.iloc[i,:][age_varName])
        Region.append(demo.iloc[i,:][race_varName])
        MCI2AD_time.append(demo.iloc[i, :][mci2ad_time_varName])


MCI_age_cluster = []
Class = []
Gender = []
Region_cluster = []
MCI2AD_time_cluster_tmp = []

num_cluster=Probability.shape[1]
#print('number of clusters: '+str(num_cluster))

for c in range(num_cluster):
    for i in range(Probability.shape[0]):
        if Probability[i].argmax() == c:
            MCI_age_cluster.append(MCI_age[i])
            Class.append(c+1)
            Gender.append(Sex[i])
            Region_cluster.append(Region[i])
            MCI2AD_time_cluster_tmp.append(MCI2AD_time[i])


Region_cluster_new = []
for region in Region_cluster:
    if region == '05' or region =='White or Caucasian': # 'White or Caucasian'
        Region_cluster_new.append('white')
    elif region == '03' or region =='Black or African American': #'Black or African American'
        Region_cluster_new.append('black')
    else:
        Region_cluster_new.append('others')

Gender_new = []
for gender in Gender:
    if gender == 'M' or gender =='Male':
        Gender_new.append('M')
    else:
        Gender_new.append('F')

dataframe = pd.DataFrame.from_dict(
    {'MCI_age_cluster': pd.Series(MCI_age_cluster),
     'Class': pd.Series(Class),
     'Gender': pd.Series(Gender_new),
     'Race': pd.Series(Region_cluster_new),
     'Progressive time': pd.Series(MCI2AD_time_cluster_tmp)
     })

dataframe.to_csv(save_path+'age_class_gender_race_PT.csv')

# zhenxing add for computing p-values----start
# compute p-value
p_value_gender_age = compute_p_value(data_df = dataframe, group ='Gender', value = 'MCI_age_cluster')
p_value_race_age = compute_p_value(data_df = dataframe, group ='Race', value = 'MCI_age_cluster')
p_value_subphenotype_age = compute_p_value(data_df = dataframe, group ='Class', value = 'MCI_age_cluster')

p_value_gender_PT = compute_p_value(data_df = dataframe, group ='Gender', value = 'Progressive time')
p_value_race_PT = compute_p_value(data_df = dataframe, group ='Race', value = 'Progressive time')
p_value_subphenotype_PT = compute_p_value(data_df = dataframe, group ='Class', value = 'Progressive time')

pval_out=open(save_path+'pvalues.txt','w')
pval_out.write('Age\n')
pval_out.write('Gender: %f\n' % p_value_gender_age)
pval_out.write('Race: %f\n' % p_value_race_age)
pval_out.write('Class: %f\n' % p_value_subphenotype_age)
## compute pval for each cluster
for c in range(num_cluster):
    sub_df = dataframe.loc[dataframe['Class']==(c+1)]
    p_value_gender = compute_p_value(sub_df, group ='Gender', value = 'MCI_age_cluster')
    p_value_race = compute_p_value(sub_df, group='Race', value='MCI_age_cluster')
    pval_out.write('Class%i_Gender: %f\n' % ((c+1),p_value_gender))
    pval_out.write('Class%i_Race: %f\n' % ((c+1), p_value_race))

pval_out.write('Progressive time\n')
pval_out.write('Gender: %f\n'%p_value_gender_PT)
pval_out.write('Race: %f\n'%p_value_race_PT)
pval_out.write('Class: %f\n'%p_value_subphenotype_PT)

## compute pval for each cluster
for c in range(num_cluster):
    sub_df = dataframe.loc[dataframe['Class']==(c+1)]
    p_value_gender = compute_p_value(sub_df, group ='Gender', value = 'Progressive time')
    p_value_race = compute_p_value(sub_df, group='Race', value='Progressive time')
    pval_out.write('Class%i_Gender: %f\n' % ((c+1),p_value_gender))
    pval_out.write('Class%i_Race: %f\n' % ((c+1), p_value_race))

pval_out.close()

plt.figure(figsize=(5,6))
sns.boxplot(x="Gender", y="MCI_age_cluster",  data=dataframe, palette="rocket", order=['M','F'])
plt.savefig(plot_save_path+'1.png')

plt.figure(figsize=(5,6))
sns.boxplot(x="Race", y="MCI_age_cluster",  data=dataframe, palette="viridis", order=['white','black','others'])
plt.savefig(plot_save_path+'2.png')

plt.figure(figsize=(5,6))
sns.boxplot(x="Class", y="MCI_age_cluster",  data=dataframe, palette="Set3", order=[1,2,3,4,5])
plt.savefig(plot_save_path+'3.png')

plt.figure(figsize=(5,6))
sns.boxplot(x="Class", y="MCI_age_cluster", hue="Gender",  data=dataframe, palette="rocket",hue_order=['M','F'])
plt.legend(loc='upper right')
plt.savefig(plot_save_path+'4.png')

plt.figure(figsize=(5,6))
sns.boxplot(x="Class", y="MCI_age_cluster", hue="Race",  data=dataframe, palette="viridis",hue_order=['white','black','others'])
plt.legend(loc='upper right')
plt.savefig(plot_save_path+'5.png')

plt.figure(figsize=(5,6))
sns.boxplot(x="Gender", y="Progressive time",  data=dataframe, palette="rocket", order=['M','F'])
plt.savefig(plot_save_path+'6.png')

plt.figure(figsize=(5,6))
sns.boxplot(x="Race", y="Progressive time",  data=dataframe, palette="viridis", order=['white','black','others'])
plt.savefig(plot_save_path+'7.png')

plt.figure(figsize=(5,6))
sns.boxplot(x="Class", y="Progressive time",  data=dataframe, palette="Set3",order=[1,2,3,4,5])
plt.savefig(plot_save_path+'8.png')

plt.figure(figsize=(5,6))
sns.boxplot(x="Class", y="Progressive time", hue="Gender",  data=dataframe, palette="rocket",hue_order=['M','F'])
plt.legend(loc='upper right')
plt.savefig(plot_save_path+'9.png')

plt.figure(figsize=(5,6))
sns.boxplot(x="Class", y="Progressive time", hue="Race",  data=dataframe, palette="viridis",hue_order=['white','black','others'])
plt.legend(loc='upper right')
plt.savefig(plot_save_path+'10.png')

