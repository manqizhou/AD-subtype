###### oneFlorida ####
#python run_all.py --version=excludeRelatedDiag \
#                  --temp_folder=Temp_data_excludeRelatedDiag \
#                  --is_train_model \
#                  --raw_diag_file_path=./full_data/DIAGNOSIS.csv \
#                  --raw_diag_varNames='["PATID","ADMIT_DATE","DX_TYPE",[9,"09","9"],[10,"10"],"DX"]' \
#                  --raw_medicine_file_path='["./full_data/DISPENSING.csv","./full_data/PRESCRIBING.csv"]' \
#                  --raw_medicine_varNames='["PATID","DATE","TYPE",["NDC"],["RxNorm"],"CODE"]' \
#                  --optimal_topic=30 \
#                  --cluster_name='v1'

## replicate clustering
python run_all.py --version=excludeRelatedDiag \
                  --temp_folder=Temp_data_excludeRelatedDiag \
                  --raw_diag_file_path=./full_data/DIAGNOSIS.csv \
                  --raw_diag_varNames='["PATID","ADMIT_DATE","DX_TYPE",[9,"09","9"],[10,"10"],"DX"]' \
                  --raw_medicine_file_path='["./full_data/DISPENSING.csv","./full_data/PRESCRIBING.csv"]' \
                  --raw_medicine_varNames='["PATID","DATE","TYPE",["NDC"],["RxNorm"],"CODE"]' \
                  --optimal_topic=30 \
                  --cluster_name='v2'

python run_all.py --version=excludeRelatedDiag \
                  --temp_folder=Temp_data_excludeRelatedDiag \
                  --raw_diag_file_path=./full_data/DIAGNOSIS.csv \
                  --raw_diag_varNames='["PATID","ADMIT_DATE","DX_TYPE",[9,"09","9"],[10,"10"],"DX"]' \
                  --raw_medicine_file_path='["./full_data/DISPENSING.csv","./full_data/PRESCRIBING.csv"]' \
                  --raw_medicine_varNames='["PATID","DATE","TYPE",["NDC"],["RxNorm"],"CODE"]' \
                  --optimal_topic=30 \
                  --cluster_name='v3'

python run_all.py --version=excludeRelatedDiag \
                  --temp_folder=Temp_data_excludeRelatedDiag \
                  --raw_diag_file_path=./full_data/DIAGNOSIS.csv \
                  --raw_diag_varNames='["PATID","ADMIT_DATE","DX_TYPE",[9,"09","9"],[10,"10"],"DX"]' \
                  --raw_medicine_file_path='["./full_data/DISPENSING.csv","./full_data/PRESCRIBING.csv"]' \
                  --raw_medicine_varNames='["PATID","DATE","TYPE",["NDC"],["RxNorm"],"CODE"]' \
                  --optimal_topic=30 \
                  --cluster_name='v4'

python run_all.py --version=excludeRelatedDiag \
                  --temp_folder=Temp_data_excludeRelatedDiag \
                  --raw_diag_file_path=./full_data/DIAGNOSIS.csv \
                  --raw_diag_varNames='["PATID","ADMIT_DATE","DX_TYPE",[9,"09","9"],[10,"10"],"DX"]' \
                  --raw_medicine_file_path='["./full_data/DISPENSING.csv","./full_data/PRESCRIBING.csv"]' \
                  --raw_medicine_varNames='["PATID","DATE","TYPE",["NDC"],["RxNorm"],"CODE"]' \
                  --optimal_topic=30 \
                  --cluster_name='v5'
