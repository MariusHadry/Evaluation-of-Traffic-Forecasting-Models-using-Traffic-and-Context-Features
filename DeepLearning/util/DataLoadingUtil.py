import gc
import math

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import zipfile
import datetime
import pandas as pd
import numpy as np

from pathlib import Path
from util import Config
from util.TrafficDataSet import TrafficDataset
from util.nbeatsx.ts_dataset import TimeSeriesDataset


def load_raw_data(p=0.10, days=30, resampling_str=None, invert_segments=False):
    """

    :param p: percentage of segments to be loaded
    :param days: number of days to be loaded
    :param resampling_str: resampling parameter
    :param invert_segments: if set to true, instead of loading the segments defined in p, the remaining segments except p are returned.
    :return:
    """

    compression_config = {'method': 'gzip'}

    if invert_segments:
        path_name = f"{Config.preprocessed_datasets}/days={days}_p={p}_resampling={resampling_str}_invert=True.pkl"
    else:
        path_name = f"{Config.preprocessed_datasets}/days={days}_p={p}_resampling={resampling_str}.pkl"

    if Path(path_name).exists() and Path(path_name).is_file():
        dataset = pd.read_pickle(path_name, compression=compression_config)
        # print(f'memory usage: {_get_memory_usage_mb(dataset)}MB')
        return dataset

    number_of_loaded_days = days
    mask = None

    with zipfile.ZipFile(Config.raw_dataset_path) as input_zip:
        sorted = input_zip.namelist().copy()
        sorted.sort()
        sorted = sorted[1:]
        start_date = sorted[0].split("/")[1].split(".")[0]
        end_date = sorted[-1].split("/")[1].split(".")[0]

        start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_date_obj = datetime.datetime.strptime(end_date, '%Y-%m-%d')

        counter = 0
        loaded_dataframe_list = []

        while start_date_obj <= end_date_obj:
            file_path = f"data/{start_date_obj.strftime('%Y-%m-%d')}.pkl"
            counter = counter + 1

            # files are contained within a zip-file. The zip-file is decompressed in-memory.
            # check if the file is contained in the zip-file
            if file_path in input_zip.namelist():
                # open the file contained in the zip-file
                with input_zip.open(file_path) as file:
                    # load file from disk into a dataframe
                    loaded_df = pickle.load(file)

                    # create filter mask
                    if mask is None:
                        np.random.seed(42)
                        mask = np.random.choice(a=[True, False], size=loaded_df.olr_code.unique().size, p=[p, 1 - p])

                        if invert_segments:
                            mask = np.logical_not(mask)

                        all_olr_codes = loaded_df.olr_code.unique()
                        filtered_olr_codes = all_olr_codes[mask]

                    # select only a specified percentage of street-segments
                    loaded_df = loaded_df.loc[loaded_df['olr_code'].isin(filtered_olr_codes)]
                    loaded_dataframe_list.append(loaded_df)

            if counter == number_of_loaded_days:
                break

            start_date_obj = start_date_obj + datetime.timedelta(days=1)

    assert (days == len(loaded_dataframe_list)), f"{days} were expected but only " \
                                                 f"{len(loaded_dataframe_list)} days were loaded!"

    dataset = pd.concat(loaded_dataframe_list, ignore_index=True)
    dataset = dataset.fillna(0)
    dataset = dataset.drop_duplicates(subset=['olr_code', 'timestamp'], keep='first', ignore_index=True)

    dataset["month"] = dataset["timestamp"].dt.month
    dataset["weekday"] = dataset["timestamp"].dt.weekday
    dataset["hour"] = dataset["timestamp"].dt.hour

    dataset["inc_type"] = dataset["inc_type"].astype(str).astype("category")
    dataset["holiday_or_vacation"] = dataset["holiday_or_vacation"].astype(int)
    dataset = dataset.drop(['has_incident'], axis=1)

    # simple mean for speed percentage
    speeds = dataset[['timestamp', 'olr_code', 'speed_percentage']].copy()

    # resample to hourly data
    speeds = speeds.set_index('timestamp').groupby('olr_code').resample(resampling_str).\
        mean(numeric_only=True).reset_index()

    dataset = pd.merge(speeds, dataset, on=['timestamp', 'olr_code'], how='left')
    dataset = dataset.fillna(method='ffill')

    dataset.drop('speed_percentage_y', axis=1, inplace=True)
    dataset.rename(columns={"speed_percentage_x": "speed_percentage"}, inplace=True)

    dataset.to_pickle(path_name, compression=compression_config)
    print(f'memory usage: {_get_memory_usage_mb(dataset)}MB')

    return dataset

def create_dataset(args, use_data_scaling=True, drop_timestamp=True):
    # Load and prepare data
    days_to_load = args['number-days-training'] + args['number-days-validation'] + args['number-days-testing']
    dataframe = load_raw_data(args['proportion-of-segments'], days_to_load, resampling_str=args['resampling-parameter'])

    dataframe = dataframe[~dataframe['olr_code'].isna()]

    dataset, args['feature_count'] = preprocess_data(dataframe, encode_time=args['encode-time'],
                                                     encode_holidays=args['encode-holidays'],
                                                     encode_incidents=args['encode-incidents'],
                                                     encode_severe_weather=args['encode-severe-weather'],
                                                     encode_segment_features=args['encode-segment-features'])
    train, val, test = split_dataset(dataset, args['number-days-training'],
                                     args['number-days-validation'], args['number-days-testing'],
                                     input_sequence_length_hours=args['window_size'])
    # free memory!
    del dataset
    gc.collect()

    if use_data_scaling:
        train, val, test, args['scalers'] = scale_data(train, val, test)

    if args['model'] != 'NBEATSx':
        train_dataset = TrafficDataset(train, train['speed_percentage'], args['window_size'],
                                       args['prediction_length'], drop_time=drop_timestamp)
        val_dataset = TrafficDataset(val, val['speed_percentage'], args['window_size'], args['prediction_length'],
                                     drop_time=drop_timestamp)
        test_dataset = TrafficDataset(test, test['speed_percentage'], args['window_size'],
                                      args['prediction_length'], drop_time=drop_timestamp)
    else:
        train_dataset = TimeSeriesDataset(X_df=train.drop(columns=['speed_percentage'], axis=1), Y_df=train[['olr_code', 'timestamp', 'speed_percentage']])
        val_dataset = TimeSeriesDataset(X_df=val.drop(columns=['speed_percentage'], axis=1), Y_df=val[['olr_code', 'timestamp', 'speed_percentage']])
        test_dataset = TimeSeriesDataset(X_df=test.drop(columns=['speed_percentage'], axis=1), Y_df=test[['olr_code', 'timestamp', 'speed_percentage']])

    return train_dataset, val_dataset, test_dataset


def split_dataset(df, num_training_days, num_val_days, num_test_days, input_sequence_length_hours=0):
    """
    Takes the first num_training_days days as training data, the next num_val_days as validation data, and the remaining
    days as test data

    :param df: dataframe that is subject to splitting
    :param num_training_days: number of days used as training data
    :param num_val_days: number of days used as validation data
    :param num_test_days: number of days used as testing data
    :return: train, val, test dataset
    """

    if input_sequence_length_hours > 0:
        input_sequence_length_hours += 1

    pd.set_option('display.max_columns', None)
    df = df.sort_values(['timestamp'])

    min_date = df['timestamp'].min().replace(hour=0, minute=0, second=0)
    max_date = df['timestamp'].max().replace(hour=23, minute=59, second=59)
    days = (max_date - min_date).days + 1

    assert (days >= num_training_days + num_val_days + num_test_days), \
        f"Split not possible! Requested split requires {num_training_days + num_val_days + num_test_days} days" \
        f" but only {days} days were given"

    train = df[df['timestamp'] < min_date + datetime.timedelta(days=num_training_days)]

    val_lower_bound = ((min_date + datetime.timedelta(days=num_training_days) -
                        datetime.timedelta(hours=input_sequence_length_hours)) < df['timestamp'])
    val_upper_bound = (df['timestamp'] < (min_date + datetime.timedelta(days=num_training_days + num_val_days)))
    val = df[val_lower_bound & val_upper_bound]

    test = df[min_date + datetime.timedelta(days=num_training_days + num_val_days) -
              datetime.timedelta(hours=input_sequence_length_hours) < df['timestamp']]

    train = train.sort_values(['olr_code', 'timestamp'])
    val = val.sort_values(['olr_code', 'timestamp'])
    test = test.sort_values(['olr_code', 'timestamp'])

    return train, val, test


def scale_data(training_data, validation_data, test_data):
    scalers = {}

    # scale target value
    scalers['speed_percentage'] = StandardScaler()

    # fit scaler on training data
    scalers['speed_percentage'].fit(training_data[['speed_percentage']])

    # transform data based on fitted scaler
    training_data['speed_percentage'] = scalers['speed_percentage'].transform(
        training_data[['speed_percentage']])
    validation_data['speed_percentage'] = scalers['speed_percentage'].transform(
        validation_data[['speed_percentage']])
    test_data['speed_percentage'] = scalers['speed_percentage'].transform(test_data[['speed_percentage']])

    # Min-Max-scaling
    for col_name in ['fow', 'frc', 'lanes', 'visibility_constraints', 'slippery_roads']:
        # check if feature is present and scale if it is the case
        if col_name in training_data.columns:
            scalers[col_name] = MinMaxScaler()
            scalers[col_name].fit(training_data[[col_name]])

            training_data[col_name] = scalers[col_name].transform(training_data[[col_name]])
            validation_data[col_name] = scalers[col_name].transform(validation_data[[col_name]])
            test_data[col_name] = scalers[col_name].transform(test_data[[col_name]])

    return training_data, validation_data, test_data, scalers


def preprocess_data(dataframe: pd.DataFrame, encode_time=False, encode_holidays=False, encode_severe_weather=False,
                    encode_incidents=False, encode_segment_features=False):
    # we always have speed percentage as feature, hence starting at 1
    feature_counter = 1

    if encode_time:
        # encode time features
        # extract hour and weekday
        dataframe["weekday"] = dataframe["timestamp"].dt.weekday.astype(int)
        dataframe["hour"] = dataframe["timestamp"].dt.hour.astype(int)

        # Normalize values to match with the 0-2π cycle
        weekday_count = 7
        hour_count = 24
        dataframe["weekday-normalized"] = 2 * math.pi * dataframe["weekday"] / weekday_count
        dataframe["hour-normalized"] = 2 * math.pi * dataframe["hour"] / hour_count

        # calculate features
        dataframe['weekday-sin'] = np.sin(dataframe["weekday-normalized"])
        dataframe['weekday-cos'] = np.cos(dataframe["weekday-normalized"])
        dataframe['hour-sin'] = np.sin(dataframe["hour-normalized"])
        dataframe['hour-cos'] = np.cos(dataframe["hour-normalized"])

        # drop temporary columns
        dataframe.drop(['weekday-normalized', 'hour-normalized'], axis=1, inplace=True)
        feature_counter += 4

    # weekday and hour are always part of the dataframes!
    dataframe.drop(['weekday', 'hour'], axis=1, inplace=True)

    # copy holiday information, no scaling required
    if encode_holidays:
        feature_counter += 1
    else:
        dataframe.drop(['holiday_or_vacation'], axis=1, inplace=True)

    if encode_severe_weather:
        # 	"t0": thunderstorm - 4 levels (max = 5)
        # 	"t1": windgusts - 4 levels
        # 	"t2": persistent rain - 4 levels
        # 	"t3": snow - 4 levels
        # 	"t4": fog (visibility <150m) - 1 level (max = 1)
        # 	"t6": icy surfaces - 3 levels (max = 4)
        #
        # 	-- excluded:
        #     	"t5": frost (should be included in icy surfaces)
        #       "t7": thaw - 3 levels

        # set higher importance for fog as there is only one level!
        dataframe.loc[dataframe['w_type_4'] == 1, 'visibility_constraints'] = 4
        dataframe['visibility_constraints'] = dataframe[['w_type_0', 'w_type_2', 'w_type_3',
                                                         'w_type_4']].max(axis=1)

        dataframe['slippery_roads'] = dataframe[['w_type_0', 'w_type_1', 'w_type_2', 'w_type_3',
                                                 'w_type_6']].max(axis=1)
        feature_counter += 2

    # the raw weather data is not used!
    dataframe.drop(['w_type_0', 'w_type_1', 'w_type_2', 'w_type_3', 'w_type_4', 'w_type_5', 'w_type_6',
                    'w_type_7', 'w_type_8', 'w_type_9'],
                   axis=1, inplace=True, errors='ignore')

    if encode_incidents:
        dataframe['construction'] = 0
        dataframe.loc[dataframe['inc_type'] == 'construction', 'construction'] = 1

        dataframe['accident'] = 0
        dataframe.loc[dataframe['inc_type'] == 'accident', 'accident'] = 1

        dataframe['other_incident'] = 0
        conditions = (dataframe['inc_type'] != 0) & (dataframe['inc_type'] != 'accident') & (
                dataframe['inc_type'] != 'construction')
        dataframe.loc[conditions, 'other_incident'] = 1

        feature_counter += 3

    dataframe.drop(['inc_type'], axis=1, inplace=True)

    if encode_segment_features:
        # move unknown value to the end of the scale
        dataframe.loc[dataframe['fow'] == 0, 'fow'] = 8
        # make scale from 0 to 7 again
        dataframe['fow'] -= 1

        feature_counter += 3
    else:
        dataframe.drop(['fow', 'frc', 'lanes'], axis=1, inplace=True)

    dataframe.drop(['maxspeed', 'length_meter', 'confidence', 'jamfactor', 'inc_criticality', 'inc_road_closed', 'month'],
                   axis=1, inplace=True)

    return dataframe, feature_counter


def _get_memory_usage_mb(dataframe: pd.DataFrame) -> float:
    return dataframe.memory_usage(deep=True).sum() / 1000.0 / 1000.0

