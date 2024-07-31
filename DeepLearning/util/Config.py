
save_to_mongodb = True
mongodb_url = 'mongodb://localhost:27080/'
mongodb_db_name = 'ray-prod-new'
mongodb_collection_name = 'simpleLSTM'
mongo_db_user = '<user>'
mongo_db_password = '<pw>'

model_name = 'simpleLSTM'   # e in ['simpleLSTM', 'simpleGRU', 'stackedLSTM', 'stackedGRU', 'NBEATS', 'NBEATSx']

device = 'cuda'         # this is only used during the evaluation!
save_to_files = False

raw_dataset_path = r"<path>/data_until_mid_june_fixed_vac.zip"
preprocessed_datasets = "<path>/preprocessed"
