"""
Evaluation scripts for ACSOS / APIN Evaluation

"""

import torch
import os
import pandas as pd
from pytorch_lightning.utilities.model_summary import ModelSummary
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.BaselineRegressor import BaselineRegressor
from util.DataLoadingUtil import load_raw_data, split_dataset, preprocess_data, create_dataset
from util.Evaluation import evaluate_instantiated_model
from util.MongoDb import load_collection, load_model, update_document, load_first, remove_new_metrics
from util.TrafficDataSet import TrafficDatasetNoSkip
from util.nbeatsx.ts_dataset import TimeSeriesDataset

from models.StackedLSTMRegressor import StackedLSTMRegressor
from models.LSTMRegressor import LSTMRegressor
from models.nbeats import NBeatsNet
from models.nbeatsx.nbeats_model import NBeatsX

import matplotlib

matplotlib.use('Agg')

base_results_dir: str = 'results_val_r2'

def _create_test_dataset(args, use_data_scaling=True, drop_timestamp=True, last_time_step='2022-06-14 23:30:00',
                         first_time_step='2022-05-29 00:30:00', scalers=None, invert_segments=False,
                         use_val=False):
    # Load and prepare data
    days_to_load = args['number-days-training'] + args['number-days-validation'] + args['number-days-testing']
    dataframe = load_raw_data(args['proportion-of-segments'], days_to_load, resampling_str=args['resampling-parameter'],
                              invert_segments=invert_segments)

    dataframe = dataframe[~dataframe['olr_code'].isna()]

    dataset, args['feature_count'] = preprocess_data(dataframe, encode_time=args['encode-time'],
                                                     encode_holidays=args['encode-holidays'],
                                                     encode_incidents=args['encode-incidents'],
                                                     encode_severe_weather=args['encode-severe-weather'],
                                                     encode_segment_features=args['encode-segment-features'])
    train, val, test = split_dataset(dataset, args['number-days-training'],
                                     args['number-days-validation'], args['number-days-testing'],
                                     input_sequence_length_hours=args['window_size'])

    raw_data = val if use_val else test

    if use_data_scaling and scalers is not None:
        raw_data['speed_percentage'] = scalers['speed_percentage'].transform(raw_data[['speed_percentage']])

        for col_name in ['fow', 'frc', 'lanes', 'visibility_constraints', 'slippery_roads']:
            # check if feature is present and scale if it is the case
            if col_name in raw_data.columns:
                raw_data[col_name] = scalers[col_name].transform(raw_data[[col_name]])

    raw_data = raw_data[(raw_data['timestamp'] >= first_time_step)]
    raw_data = raw_data[(raw_data['timestamp'] <= last_time_step)]

    if drop_timestamp:
        # drop timestamp from data
        raw_data = raw_data.drop(['timestamp'], axis=1)

    if args['model'] == 'NBEATSx':
        olr_codes = raw_data['olr_code'].unique()
        dataset_list = []

        for o in olr_codes:
            tmp = raw_data[raw_data['olr_code'] == o]
            tmp_ds = TimeSeriesDataset(X_df=tmp.drop(columns=['speed_percentage'], axis=1),
                                       Y_df=tmp[['olr_code', 'timestamp', 'speed_percentage']])
            dataset_list.append(tmp_ds)

        return dataset_list

    return TrafficDatasetNoSkip(raw_data, raw_data['speed_percentage'], args['window_size'],
                                args['prediction_length'])


def _get_olr_codes(args, last_time_step='2022-06-14 23:30:00'):
    # Load and prepare data
    days_to_load = args['number-days-training'] + args['number-days-validation'] + args['number-days-testing']
    dataframe = load_raw_data(args['proportion-of-segments'], days_to_load, resampling_str=args['resampling-parameter'])

    dataframe = dataframe[~dataframe['olr_code'].isna()]

    dataset, args['feature_count'] = preprocess_data(dataframe, encode_time=args['encode-time'],
                                                     encode_holidays=args['encode-holidays'],
                                                     encode_incidents=args['encode-incidents'],
                                                     encode_severe_weather=args['encode-severe-weather'],
                                                     encode_segment_features=args['encode-segment-features'])
    _, _, test = split_dataset(dataset, args['number-days-training'],
                               args['number-days-validation'], args['number-days-testing'],
                               input_sequence_length_hours=args['window_size'])
    test = test[(test['timestamp'] <= last_time_step)]

    print(len(test['olr_code'].unique()))
    tmp = test['olr_code'].unique()
    pd.DataFrame(tmp).to_csv(f"{args['resampling-parameter']}_olrs_dl.csv")


def _get_model_class(model_name):
    if model_name == 'LSTMRegressor':
        return LSTMRegressor
    elif model_name == 'StackedLSTMRegressor':
        return StackedLSTMRegressor
    elif model_name == 'NBEATS':
        return NBeatsNet
    elif model_name == 'NBEATS-trend-seasonality':
        return NBeatsNet
    elif model_name == 'NBEATSx':
        return NBeatsX


def reset_new_metrics(database_name):
    collection_names = ['simpleLSTM', 'simpleGRU', 'stackedLSTM', 'stackedGRU', 'NBEATS', 'NBEATS-trend-seasonality',
                        'NBEATSx']

    for collection_name in collection_names:
        remove_new_metrics(database_name, collection_name)


def reevaluate(database_name, invert_segments=False):
    """
    Calculate metrics for models based on different testing period. Testing period is ensured in
    `_create_test_dataset()`.

    :param database_name: Name of the database that should be reevaluated (ray-prod-05H, ray-prod-new).
    :return:
    """
    collection_names = ['NBEATSx', 'simpleLSTM', 'simpleGRU', 'stackedLSTM', 'stackedGRU', 'NBEATS',
                        'NBEATS-trend-seasonality']

    field_name = "metrics_inverted_segments" if invert_segments else "metrics_new"

    for collection_name in collection_names:
        collection_list = load_collection(database_name=database_name, collection_name=collection_name)

        for entry in tqdm(collection_list, desc=f'{collection_name} progress'):
            if field_name in entry:
                continue

            # get model from MongoDB
            model, scalers = load_model(database_name, entry)
            model_class = _get_model_class(entry['experiment_config']['model'])

            # load evaluation data
            if entry['experiment_config']['model'] != 'NBEATSx':
                test_dataset = _create_test_dataset(entry['experiment_config'], scalers=scalers,
                                                    invert_segments=invert_segments)
                metrics = evaluate_instantiated_model(entry['experiment_config'], model, scalers, test_dataset,
                                                      model_class)
            else:
                test_dataset_list = _create_test_dataset(entry['experiment_config'], scalers=scalers,
                                                         invert_segments=invert_segments, drop_timestamp=False)
                metrics = evaluate_instantiated_model(entry['experiment_config'], model, scalers, None,
                                                      model_class, test_dataset_list=test_dataset_list)

            to_insert = {
                field_name: metrics
            }

            # insert new metrics into document
            update_document(database_name, collection_name, entry['_id'], new_data=to_insert)


def evaluate_best_models(database_name, invert_segments=False):
    results_dir = f'{base_results_dir}/{database_name}/'
    os.makedirs(results_dir, exist_ok=True)

    collection_names = ['NBEATSx', 'simpleLSTM', 'simpleGRU', 'stackedLSTM', 'stackedGRU', 'NBEATS',
                        'NBEATS-trend-seasonality']

    for collection in tqdm(collection_names, desc=f'Progress {database_name}'):
        # sort_parameters = [('metrics_new.r^2', -1)]
        sort_parameters = [('validation_metrics.r^2', -1)]
        entry = load_first(database_name, collection, sort_parameters)
        print(entry["experiment_config"])

        # get model from MongoDB
        model, scalers = load_model(database_name, entry)
        model_class = _get_model_class(entry['experiment_config']['model'])

        # load evaluation data
        if entry['experiment_config']['model'] == 'NBEATSx':
            test_dataset_list = _create_test_dataset(entry['experiment_config'], drop_timestamp=False, scalers=scalers,
                                                     invert_segments=invert_segments)
            metrics = evaluate_instantiated_model(entry['experiment_config'], model, scalers, None,
                                                  model_class, test_dataset_list=test_dataset_list,
                                                  append_predicted_values=True)
        else:
            test_dataset = _create_test_dataset(entry['experiment_config'], scalers=scalers,
                                                invert_segments=invert_segments)
            metrics = evaluate_instantiated_model(entry['experiment_config'], model, scalers, test_dataset,
                                                  model_class, append_predicted_values=True)

        metrics_df = pd.DataFrame()
        metrics_df['truth'] = metrics['truth']
        metrics_df['predicted'] = metrics['predicted']
        metrics_df.to_csv(f'{results_dir}/{collection}_inverted={invert_segments}.csv', sep=';')


def save_olr_codes(database_name):
    results_dir = f'{base_results_dir}/{database_name}/'
    os.makedirs(results_dir, exist_ok=True)

    collection = 'simpleLSTM'
    # sort_parameters = [('metrics_new.r^2', -1)]
    sort_parameters = [('validation_metrics.r^2', -1)]
    entry = load_first(database_name, collection, sort_parameters)
    _get_olr_codes(entry['experiment_config'])


def evaluate_hold_baseline_regressor(database_name, invert_segments=False):
    results_dir = f'{base_results_dir}/{database_name}/'
    os.makedirs(results_dir, exist_ok=True)

    # load an entry to determine test dataset..
    entry = load_first(database_name, 'simpleLSTM')

    baseline_conf = {'encode-time': False,
                     'encode-holidays': False,
                     'encode-severe-weather': False,
                     'encode-incidents': False,
                     'encode-segment-features': False}
    entry.update(baseline_conf)
    args = entry['experiment_config']

    test_dataset = _create_test_dataset(args, use_data_scaling=False, invert_segments=invert_segments)
    model = BaselineRegressor(args['prediction_length'])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    forecast_result, truth_result = [], []

    for i, (features, truth) in enumerate(tqdm(test_loader, desc=f'{database_name} baseline-regressor:')):
        forecast = model(features)

        forecast_result.extend(forecast.view(-1).tolist())
        truth_result.extend(truth.view(-1).tolist())

    metrics_df = pd.DataFrame()
    metrics_df['truth'] = truth_result
    metrics_df['predicted'] = forecast_result
    metrics_df.to_csv(f'{results_dir}/hold-baseline_inverted={invert_segments}.csv', sep=';')


def evaluate_100p_baseline_regressor(database_name, invert_segments=False):
    """
    Evaluation of baseline regressor that always predicts 100% for the speed percentage.

    :param database_name:
    :return:
    """
    results_dir = f'{base_results_dir}/{database_name}/'
    os.makedirs(results_dir, exist_ok=True)

    # load an entry to determine test dataset..
    entry = load_first(database_name, 'simpleLSTM')

    baseline_conf = {'encode-time': False,
                     'encode-holidays': False,
                     'encode-severe-weather': False,
                     'encode-incidents': False,
                     'encode-segment-features': False}

    entry.update(baseline_conf)
    args = entry['experiment_config']

    test_dataset = _create_test_dataset(args, use_data_scaling=False, invert_segments=invert_segments)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    forecast_result, truth_result = [], []

    for i, (features, truth) in enumerate(tqdm(test_loader, desc=f'{database_name} 100p-regressor:')):
        forecast = [1.0 for _ in range(0, truth.shape[1])]
        forecast_result.extend(forecast)
        truth_result.extend(truth.view(-1).tolist())

    metrics_df = pd.DataFrame()
    metrics_df['truth'] = truth_result
    metrics_df['predicted'] = forecast_result
    metrics_df.to_csv(f'{results_dir}/100p-baseline_inverted={invert_segments}.csv', sep=';')


def get_best_model_sizes():
    collection_names = ['NBEATS', 'NBEATSx', 'simpleGRU', 'simpleLSTM', 'stackedGRU', 'stackedLSTM']
    data = []

    models_05h_path = f'model_pkls/0.5h/'
    models_1h_path = f'model_pkls/1h/'
    os.makedirs(models_05h_path, exist_ok=True)
    os.makedirs(models_1h_path, exist_ok=True)

    for collection in tqdm(collection_names, desc=f'Progress'):
        sort_parameters = [('validation_metrics.r^2', -1)]

        # get 1h model from MongoDB
        entry = load_first('ray-prod-new', collection, sort_parameters)
        model_1h, _ = load_model('ray-prod-new', entry)
        summary_1h = ModelSummary(model_1h)
        torch.save(model_1h, f'{models_1h_path}/{collection}.pkl')

        # get .5h model from MongoDB
        entry = load_first('ray-prod-05H', collection, sort_parameters)
        model_05h, _ = load_model('ray-prod-05H', entry)
        summary_05h = ModelSummary(model_05h)
        torch.save(model_05h, f'{models_05h_path}/{collection}.pkl')

        data.append(
            {
                'Model Name': collection,
                'Trainable Parameters (1h)': summary_1h.trainable_parameters,
                'Estimated Size (MB) (1h)': summary_1h.model_size,
                'Trainable Parameters (0.5h)': summary_05h.trainable_parameters,
                'Estimated Size (MB) (0.5h)': summary_05h.model_size,
            }
        )

    df = pd.DataFrame(data)
    df.reset_index(drop=True)
    return df.to_latex(index=False)


def calculate_val_metrics(database_name):
    collection_names = ['NBEATSx', 'simpleLSTM', 'simpleGRU', 'stackedLSTM', 'stackedGRU', 'NBEATS',
                        'NBEATS-trend-seasonality']
    field_name = "validation_metrics"

    for collection_name in collection_names:
        collection_list = load_collection(database_name=database_name, collection_name=collection_name)

        for entry in tqdm(collection_list, desc=f'{collection_name} progress'):
            if field_name in entry:
                continue

            # get model from MongoDB
            model, scalers = load_model(database_name, entry)
            model_class = _get_model_class(entry['experiment_config']['model'])

            _first_time_step = '2022-05-08 00:30:00'  # 09.05.2022
            _last_time_step = '2022-05-29 23:30:00'  # 30.05.2022

            # load evaluation data
            if entry['experiment_config']['model'] != 'NBEATSx':
                val_dataset = _create_test_dataset(entry['experiment_config'], scalers=scalers, use_val=True,
                                                   first_time_step=_first_time_step, last_time_step=_last_time_step)
                metrics = evaluate_instantiated_model(entry['experiment_config'], model, scalers, val_dataset,
                                                      model_class)
            else:
                val_dataset_list = _create_test_dataset(entry['experiment_config'], scalers=scalers, use_val=True,
                                                        drop_timestamp=False,
                                                        first_time_step=_first_time_step, last_time_step=_last_time_step)
                metrics = evaluate_instantiated_model(entry['experiment_config'], model, scalers, None,
                                                      model_class, test_dataset_list=val_dataset_list)

            to_insert = {
                field_name: metrics
            }

            # insert new metrics into document
            update_document(database_name, collection_name, entry['_id'], new_data=to_insert)


def extract_best_configs():
    collection_names = ['NBEATSx', 'simpleLSTM', 'simpleGRU', 'stackedLSTM', 'stackedGRU', 'NBEATS']
    dict_list = []

    for db in ['ray-prod-new', 'ray-prod-05H']:
        for collection in collection_names:
            sort_parameters = [('validation_metrics.r^2', -1)]
            doc = load_first(db, collection, sort_parameters)

            to_add = {
                'db': db,
                'collection': collection,
                'primary': doc['experiment_config']['primary_features'],
                'secondary': doc['experiment_config']['secondary_features'],
                'batch_size': doc['experiment_config']['batch_size'],
                'learning_rate': doc['experiment_config']['learning_rate']
            }

            to_add = {**to_add, **doc['model']['model_hparams']}
            dict_list.append(to_add)

    os.makedirs("../evaluationOutput/", exist_ok=True)
    df = pd.DataFrame(dict_list)
    df.to_csv("../evaluationOutput/configs.csv", sep=";")



if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '10'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    calculate_val_metrics('ray-prod-new')
    calculate_val_metrics('ray-prod-05H')

    evaluate_best_models('ray-prod-new')
    evaluate_best_models('ray-prod-05H')

    print(get_best_model_sizes())   # print LaTeX table
    extract_best_configs()  # saves to *.csv

    # baselines
    evaluate_hold_baseline_regressor('ray-prod-new')
    evaluate_hold_baseline_regressor('ray-prod-05H')
    evaluate_100p_baseline_regressor('ray-prod-new')
    evaluate_100p_baseline_regressor('ray-prod-05H')

    # check inverted segments!
    evaluate_best_models('ray-prod-new', invert_segments=True)
    evaluate_best_models('ray-prod-05H', invert_segments=True)

    # baselines for inverted segments
    evaluate_hold_baseline_regressor('ray-prod-new', invert_segments=True)
    evaluate_hold_baseline_regressor('ray-prod-05H', invert_segments=True)
    evaluate_100p_baseline_regressor('ray-prod-new', invert_segments=True)
    evaluate_100p_baseline_regressor('ray-prod-05H', invert_segments=True)
