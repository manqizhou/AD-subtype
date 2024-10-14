import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.stats import chi2_contingency
from itertools import combinations
from statsmodels.sandbox.stats.multicomp import multipletests
import scikit_posthocs as sp
import sys
import os
######### load data
stratified_path=sys.argv[1]#'./results/stratified_analysis/'
save_path=sys.argv[2]

if not os.path.exists(save_path):
    os.makedirs(save_path)

age_class_gender_race_PT=pd.read_csv(stratified_path+'age_class_gender_race_PT.csv')
out=open(save_path+'pairwise_pval.tsv','w')

#################### get pvalues
num_class=len(age_class_gender_race_PT.Class.unique())
S = ['S'+str(i+1) for i in range(num_class)]#class

## Sex, Region: chi square test, FDR correction
def pvalue_sex_race(df,out,num_class):
    chi2, p, dof, ex = chi2_contingency(df.iloc[:, 1:], correction=True)
    out.write(f"Chi2 result of the contingency table: {chi2}, p-value: {p}\n")
    all_combinations = list(combinations(df.index, 2))
    p_vals = []
    A=np.full((num_class, num_class), np.nan)
    for comb in all_combinations:
        # subset df into a dataframe containing only the pair "comb"
        new_df = df[(df.index == comb[0]) | (df.index == comb[1])]
        # running chi2 test
        chi2, p, dof, ex = chi2_contingency(new_df.iloc[:, 1:], correction=True)
        p_vals.append(p)
    reject_list, corrected_p_vals = multipletests(p_vals, method='fdr_bh')[:2]
    #out.write("comb\toriginal p-value\tcorrected p-value\treject?\n")
    for comb, p_val, corr_p_val, reject in zip(all_combinations, p_vals, corrected_p_vals, reject_list):
        A[comb[0],comb[1]]=corr_p_val
        #out.write(str(comb)+"\t"+str(p_val)[:5]+"\t"+str(corr_p_val)[:5]+"\t"+str(reject)+'\n')
    A=pd.DataFrame(A)
    out.write(str(A) + '\n')

## region
out.write('!! Race !!\n')

white = list(age_class_gender_race_PT[age_class_gender_race_PT.Race=='white'].Class.value_counts(sort=False))
black = list(age_class_gender_race_PT[age_class_gender_race_PT.Race=='black'].Class.value_counts(sort=False))
others = list(age_class_gender_race_PT[age_class_gender_race_PT.Race=='others'].Class.value_counts(sort=False))

df = pd.DataFrame.from_dict(
    {'S': S,
     'White': white,
     'Black': black,
     'Others': others}
)
pvalue_sex_race(df,out,num_class)

## sex
out.write('!! sex !!\n')
male = list(age_class_gender_race_PT[age_class_gender_race_PT.Gender=='M'].Class.value_counts(sort=False))
female = list(age_class_gender_race_PT[age_class_gender_race_PT.Gender=='F'].Class.value_counts(sort=False))
df = pd.DataFrame.from_dict(
    {'S': S,
     'Female': female,
     'Male': male}
)
pvalue_sex_race(df,out,num_class)

## age, PT : Kruskal-Wallis test

## PT
out.write('!! PT !!\n')
data = [age_class_gender_race_PT.loc[ids, 'Progressive time'].values for ids in age_class_gender_race_PT.groupby('Class').groups.values()]
H, p = ss.kruskal(*data)
out.write(f"KW result of PT on different subphenotypes, p-value: {p}\n")
A = sp.posthoc_dunn(age_class_gender_race_PT, val_col='Progressive time', group_col='Class')
out.write(str(A)+'\n')

## age
out.write('!! age !!\n')
data = [age_class_gender_race_PT.loc[ids, 'MCI_age_cluster'].values for ids in age_class_gender_race_PT.groupby('Class').groups.values()]
H, p = ss.kruskal(*data)
out.write(f"KW result of Age on different subphenotypes, p-value: {p}\n")
A = sp.posthoc_dunn(age_class_gender_race_PT, val_col='MCI_age_cluster', group_col='Class', p_adjust='fdr_bh')
out.write(str(A)+'\n')

out.close()
