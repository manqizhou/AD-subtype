import os
import argparse
import os.path

parser = argparse.ArgumentParser(description="Run dynaPheM and downstreat analysis")
parser.add_argument("--version", help="topic modeling version name", required=True, type = str)
parser.add_argument("--temp_folder", help="temp_folder name", required=True, type = str)
parser.add_argument("--raw_diag_file_path", help="raw diagnosis file path", required=True, type = str)
parser.add_argument("--raw_diag_varNames", help="variable names representing [patient id, date, code type, [ICD9 type names], [ICD10 type names], code]", required=True, type = str)
parser.add_argument("--raw_medicine_file_path", help="raw medicine file path", required=True, type = str)
parser.add_argument("--raw_medicine_varNames", help="variable names representing [patient id, date, code type, [NDC type names], [RXNORM type names], code]", required=True, type = str)
parser.add_argument("--optimal_topic", help="optimal topic number", required=True, metavar=30, type = int)
parser.add_argument("--is_train_model", help="whether to train topic modeling", required=False, action='store_true')

parser.add_argument("--num_cluster", help="number of cluster", required=False, default=5, type = int)
parser.add_argument("--top_num", help="number of saved top features", required=False, default=10,type = int)
parser.add_argument("--cluster_name", help="cluster version name", required=False, default='v1', type = str)
parser.add_argument("--min_topic", help="minimum topic number", required=False, default=10, type = int)
parser.add_argument("--max_topic", help="maximum topic number", required=False, default=50, type = int)

args = parser.parse_args()

if args.is_train_model:
    assert args.min_topic, "Provide --min_topic"
    assert args.max_topic, "Provide --max_topic"

#################### STEP 0: finish the data_preparation part
version=args.version#'excludeRelatedDiag'#topic model version name to store results
temp_folder=args.temp_folder#'Temp_data_excludeRelatedDiag'# temp data folder name
raw_diag_file_path=args.raw_diag_file_path
raw_diag_varNames=args.raw_diag_varNames
raw_medicine_file_path=args.raw_medicine_file_path
raw_medicine_varNames=args.raw_medicine_varNames

optimal_topic='topic'+str(args.optimal_topic)#for UCSF, change to topic20
num_cluster = args.num_cluster#5 # stick to 5 clusters
min_topic=args.min_topic#10
max_topic=args.max_topic#50# can set max_topic=min_topic if only run one setting
top_num=args.top_num#10# save top top_num features
cluster_name=args.cluster_name
is_train_model=args.is_train_model

print('read parameters successfully')
drug_rxnorm_codes_f='./drug_rxnorm_codes.pickle'#medicine ref file from Alice

# ####### change to your name before start
# # raw diagnosis file path
# raw_diag_file_path='./full_data/DIAGNOSIS.csv'
#
# '''
# variable names in the raw diagnosis file representing [patient id, date, code type, ICD9 type names, ICD10 type names, code]
# oneFlorida: '\'["PATID","ADMIT_DATE","DX_TYPE",[9,"09","9"],[10,"10"],"DX"]\''
# UCSF: '\'["PatientID","StartDate","Type",["ICD-9-CM"],["ICD-10-CM"],"Value"]\''
# INSIGHT: []
# '''
# raw_diag_varNames='\'["PATID","ADMIT_DATE","DX_TYPE",[9,"09","9"],[10,"10"],"DX"]\''#oneFlorida
#
# # raw medicine file
# raw_medicine_file_path='\'["./full_data/DISPENSING.csv","./full_data/PRESCRIBING.csv"]\''#raw prescribing files, used in count medicine in each cluster
#
# '''
# variable names in the raw diagnosis file representing [patient id, date, code type, NDC type names, RXNORM type names, code]
# oneFlorida: '\'["PATID",["DISPENSE_DATE","RX_ORDER_DATE"],["NDC","RXNORM_CUI"],["NDC"],["RxNorm"],["NDC","RXNORM_CUI"]]\''
# UCSF: '\'["PatientID","OrderDate","Type",["NDC"],["RxNorm"],"Code"]\''
# INSIGHT: []
# '''
# raw_medicine_varNames='\'["PATID",["DISPENSE_DATE","RX_ORDER_DATE"],["NDC","RXNORM_CUI"],["NDC"],["RxNorm"],["NDC","RXNORM_CUI"]]\''

#################### STEP 1: run topic modeling
if is_train_model:#if train topic modeling
    os.chdir('./DMTM/python_code/')# change working directory

    temp_data_path='../../data_preparation/'+temp_folder+'/'#temp_data_path
    model_save_path='./trained_model_'+version+'/'
    result_save_path='../../results/'+version+'/'

    print('start topic modeling!')#will run parallel
    cmd='python run_topic_model.py %s %s %s %s %s' % (temp_data_path,model_save_path,result_save_path,min_topic,max_topic)
    os.system(cmd)
    print('finish topic modeling!')

    ################### STEP 2: key topic analysis
    os.chdir('../../')# change working directory

    model_path='./DMTM/python_code/trained_model_'+version+'/'+optimal_topic+'/'
    temp_data_path='./data_preparation/'+temp_folder+'/'
    mapping_path='./data_preparation/dic/'
    save_path='./results/'+version+'/'

    print('start key topic analysis!')
    #### get key topics
    cmd='python get_key_topics.py %s %s %s %s %s' % (model_path,temp_data_path,mapping_path,save_path,top_num)
    os.system(cmd)

    #### get all topics top codes
    cmd='python get_top_codes_in_topics.py %s %s %s %s' % (model_path,temp_data_path,save_path,top_num)
    os.system(cmd)

    #### get names/weights/categories in key topics
    ref_path='./data_preparation/mapping/'
    cmd='python get_name_weight_key_topic.py %s %s %s %s' % (model_path,save_path,ref_path,top_num)
    os.system(cmd)

    #### get transition
    key_topic_path='./results/'+version+'/'
    save_path='./results/'+version+'/transition/'

    cmd='python get_transition.py %s %s %s' % (model_path,key_topic_path,save_path)
    os.system(cmd)

    print('finish key topic analysis!')

else:
    key_topic_f='./results/'+version+'/key_topic.txt'
    if not os.path.isfile(key_topic_f):
        print('do not have key topic. Start getting key topic!')
        model_path='./DMTM/python_code/trained_model_'+version+'/'+optimal_topic+'/'
        temp_data_path='./data_preparation/'+temp_folder+'/'
        mapping_path='./data_preparation/dic/'
        save_path='./results/'+version+'/'
        #### get key topics
        cmd='python get_key_topics.py %s %s %s %s %s' % (model_path,temp_data_path,mapping_path,save_path,top_num)
        os.system(cmd)

        #### get all topics top codes
        cmd='python get_top_codes_in_topics.py %s %s %s %s' % (model_path,temp_data_path,save_path,top_num)
        os.system(cmd)

        #### get names/weights/categories in key topics
        ref_path='./data_preparation/mapping/'
        cmd='python get_name_weight_key_topic.py %s %s %s %s' % (model_path,save_path,ref_path,top_num)
        os.system(cmd)

        #### get transition
        key_topic_path='./results/'+version+'/'
        save_path='./results/'+version+'/transition/'

        cmd='python get_transition.py %s %s %s' % (model_path,key_topic_path,save_path)
        os.system(cmd)

        print('finish key topic analysis!')
        
    print('Do clustering on topic modeling saved under '+'../DMTM/python_code/trained_model_'+version+'/'+optimal_topic+'/')

################### STEP 3: run clustering
os.chdir('./TCLA/')
print('start clustering!')

save_path = './'+str(num_cluster)+'cluster_'+version+'/'+cluster_name+'/'
model_path='../DMTM/python_code/trained_model_'+version+'/'+optimal_topic+'/'
temp_data_path='../data_preparation/'+temp_folder+'/'
key_topic_path = '../results/'+version+'/'

cmd='python T_GBTM_main.py %s %s %s %s %s' % (num_cluster,save_path,model_path,temp_data_path,key_topic_path)
os.system(cmd)

################### STEP 4: get cluster and topic change
os.chdir('../')
temp_data_path='./data_preparation/'+temp_folder+'/'

print('start plotting cluster and topic change!')

cluster_path='./TCLA/'+str(num_cluster)+'cluster_'+version + '/' + cluster_name+'/'
model_path='./DMTM/python_code/trained_model_'+version+'/'+optimal_topic+'/'
key_topic_path='./results/'+version+'/'
save_path = './results/'+version + '/' + cluster_name+'/cluster_topic_change/'

cmd='python get_topic_cluster_change.py %s %s %s %s' % (cluster_path,model_path,key_topic_path,save_path)
os.system(cmd)

cmd='python plot_diag_change_histogram.py %s %s %s %s' % (temp_data_path,key_topic_path,cluster_path,save_path)
os.system(cmd)

print('finish plotting cluster and topic change!')


################### STEP 4: KM analysis
print('start KM analysis!')

cluster_path = './TCLA/' + str(num_cluster) + 'cluster_' + version + '/' + cluster_name + '/'
save_path='./results/'+version+'/'+ cluster_name +'/KM/'

cmd='python get_KM.py %s %s %s' % (cluster_path,temp_data_path,save_path)
os.system(cmd)

cmd='Rscript plot_KM.R %s %s' % (save_path,save_path)
os.system(cmd)

################### STEP 4: plot topic change heatmap
KM_result_path = './results/'+version + '/' + cluster_name + '/KM/'
save_path = './results/'+version + '/' + cluster_name+'/cluster_topic_change/'

cmd='python plot_diag_change_heatmap.py %s %s %s %s' % (temp_data_path,key_topic_path,KM_result_path,save_path)
os.system(cmd)

################### STEP 5: get stratified analysis
print('start stratified analysis!')

cluster_path = './TCLA/' + str(num_cluster) + 'cluster_' + version + '/' + cluster_name + '/'
save_path='./results/' + version +'/'+ cluster_name +'/stratified_analysis/'

cmd='python get_stratified_analysis.py %s %s %s' % (cluster_path,temp_data_path,save_path)
os.system(cmd)

print('finish stratified analysis!')
#
################## STEP 6: get pairwise pval
print('start pairwise analysis!')

stratified_path='./results/'+version +'/'+cluster_name +'/stratified_analysis/'
save_path='./results/'+version + '/' + cluster_name + '/pairwise_pval/'

cmd='python get_pairwise_pval.py %s %s' % (stratified_path,save_path)
os.system(cmd)

print('finish pairwise analysis!')

################### STEP 7: get cluster stats
print('start extracting cluster statistics!')
mapping_path = './data_preparation/mapping/'

KM_result_path = './results/'+version + '/' + cluster_name + '/KM/'
save_path = './results/' + version+ '/' + cluster_name + '/cluster_stats/'

### age/sex/PT
cmd='python get_cluster_stats.py %s %s %s' % (temp_data_path,KM_result_path,save_path)
os.system(cmd)

### comorbidity
cmd='''python get_cluster_comorbidity.py %s %s %s %s \'%s\' %s''' % (temp_data_path,KM_result_path,mapping_path,
                                                             raw_diag_file_path,raw_diag_varNames,save_path)

os.system(cmd)

### medicine
cmd='''python get_cluster_medicine.py %s %s %s %s \'%s\' \'%s\' %s''' % (temp_data_path, KM_result_path, mapping_path, drug_rxnorm_codes_f,
                                                             raw_medicine_file_path, raw_medicine_varNames, save_path)

os.system(cmd)

print('finish extracting cluster statistics!')

########
print('Finish All!')