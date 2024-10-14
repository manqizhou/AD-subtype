Assumed folder structure:
└── code/
    ├── data_preparation/
    │   ├── dic
    │   ├── mapping
    │   └── Temp_data_versionName # stored preprocessed data
    ├── DMTM/
    │   └── python_code/
    │       ├── pydpm
    │       └── trained_model_versionName # will store topic model resuts/
    │           ├── topic10
    │           ├── ...
    │           ├── topic30
    │           └── ...
    ├── TCLA/
    │   └── 5cluster_versionName # will store clustering results
    └── results /
        └── versionName # will store downstream analysis results

Copy your preprocessed Temp_data_versionName to the data_preparation folder.


To run topic modeling, clustering, and all downstream analysis,
python run_all.py --version=excludeRelatedDiag \
                  --temp_folder=Temp_data_excludeRelatedDiag \
                  --is_train_model \
                  --raw_diag_file_path='./full_data/DIAGNOSIS.csv' \
                  --raw_diag_varNames='["PATID","ADMIT_DATE","DX_TYPE",[9,"09","9"],[10,"10"],"DX"]' \
                  --raw_medicine_file_path='["./full_data/DISPENSING.csv","./full_data/PRESCRIBING.csv"]' \
                  --raw_medicine_varNames='["PATID","DATE","TYPE",["NDC"],["RxNorm"],"CODE"]' \
                  --optimal_topic=30 \
                  --cluster_name='v1'

Parameters explanation:
--version: model version name
--temp_folder: Temp_data folder name
--is_train_model: whether to train topic model or not.
                  If add --is_train_model, it will train 10-50 topic modeling.
                  If delete --is_train_model, it will jump to clustering
--raw_diag_file_path: file path of the original raw diagnosis file
--raw_diag_varNames: column names in the raw diagnosis file.
                     [ PATIENTID column name,
                       DATE column name,
                       TYPE column name,
                      [a list of elements in TYPE column referring to ICD9],
                      [a list of elements in TYPE column referring to ICD10],
                      CODE column name
                     ]
--raw_medicine_file_path: [file path of the original raw medicine file].
                          If has one medicine file, [file1]; if has two medicine files, [file1,file2]
--raw_diag_varNames: column names in the raw medicine file.
                     [ PATIENTID column name,
                       DATE column name,
                       TYPE column name,
                      [a list of elements in TYPE column referring to NDC],
                      [a list of elements in TYPE column referring to RxNorm],
                      CODE column name
                     ]
--optimal_topic: the optimal topic number. Clustering will only run on the optimal topic model
--cluster_name: cluster version name. Default if v1. Can change to other names if want to replicate clustering.

*NOTE* Do not change quotes in the --raw* parameter. use " for string and add '' for the whole list.
