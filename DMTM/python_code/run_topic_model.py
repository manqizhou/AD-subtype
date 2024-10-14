import sys
import os

data_path=sys.argv[1]#'../../data_preparation/Temp_data_rm_all_AD_MCI_dementia_related_diag/'
save_path=sys.argv[2]#save model, ./trained_model_exclude_related_diag/
result_path=sys.argv[3]# save topic coherence
min_topic=int(sys.argv[4])
max_topic=int(sys.argv[5])

for i in range(min_topic,max_topic+1,5):
   model_path=save_path+'topic'+str(i)+'/'
   print('run topic '+str(i))
   if i==max_topic:
      os.system('python DMTM_main.py %s %s %s' % (data_path,model_path,i))
   else:
      os.system('python DMTM_main.py %s %s %s &' % (data_path, model_path, i))

for i in range(min_topic,max_topic+1,5):
   model_path=save_path+'topic'+str(i)+'/'
   os.system('python cal_likelihood_coherence.py %s %s %s %s' % (data_path,model_path,i,result_path))
