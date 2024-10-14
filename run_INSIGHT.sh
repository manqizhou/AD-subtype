###### INSIGHT ####
python run_all.py --version=excludeRelatedDiag \
                  --temp_folder=Temp_data_excludeRelatedDiag \
                  --is_train_model \
                  --raw_diag_file_path=path/to/diagnosis.csv \
                  --raw_diag_varNames='["PatientID","StartDate","Type",["ICD-9-CM"],["ICD-10-CM"],"Value"]' \
                  --raw_medicine_file_path='["path/to/medicine.csv"]' \
                  --raw_medicine_varNames='["PatientID","OrderDate","Type",["NDC"],["RxNorm"],"Code"]' \
                  --optimal_topic=20 \
                  --cluster_name='v1'

## replicate clustering
python run_all.py --version=excludeRelatedDiag \
                  --temp_folder=Temp_data_excludeRelatedDiag \
                  --raw_diag_file_path=path/to/diagnosis.csv \
                  --raw_diag_varNames='["PatientID","StartDate","Type",["ICD-9-CM"],["ICD-10-CM"],"Value"]' \
                  --raw_medicine_file_path='["path/to/medicine.csv"]' \
                  --raw_medicine_varNames='["PatientID","OrderDate","Type",["NDC"],["RxNorm"],"Code"]' \
                  --optimal_topic=20 \
                  --cluster_name='v2' &

python run_all.py --version=excludeRelatedDiag \
                  --temp_folder=Temp_data_excludeRelatedDiag \
                  --raw_diag_file_path=path/to/diagnosis.csv \
                  --raw_diag_varNames='["PatientID","StartDate","Type",["ICD-9-CM"],["ICD-10-CM"],"Value"]' \
                  --raw_medicine_file_path='["path/to/medicine.csv"]' \
                  --raw_medicine_varNames='["PatientID","OrderDate","Type",["NDC"],["RxNorm"],"Code"]' \
                  --optimal_topic=20 \
                  --cluster_name='v3' &

python run_all.py --version=excludeRelatedDiag \
                  --temp_folder=Temp_data_excludeRelatedDiag \
                  --raw_diag_file_path=path/to/diagnosis.csv \
                  --raw_diag_varNames='["PatientID","StartDate","Type",["ICD-9-CM"],["ICD-10-CM"],"Value"]' \
                  --raw_medicine_file_path='["path/to/medicine.csv"]' \
                  --raw_medicine_varNames='["PatientID","OrderDate","Type",["NDC"],["RxNorm"],"Code"]' \
                  --optimal_topic=20 \
                  --cluster_name='v4' &

python run_all.py --version=excludeRelatedDiag \
                  --temp_folder=Temp_data_excludeRelatedDiag \
                  --raw_diag_file_path=path/to/diagnosis.csv \
                  --raw_diag_varNames='["PatientID","StartDate","Type",["ICD-9-CM"],["ICD-10-CM"],"Value"]' \
                  --raw_medicine_file_path='["path/to/medicine.csv"]' \
                  --raw_medicine_varNames='["PatientID","OrderDate","Type",["NDC"],["RxNorm"],"Code"]' \
                  --optimal_topic=20 \
                  --cluster_name='v5' &
