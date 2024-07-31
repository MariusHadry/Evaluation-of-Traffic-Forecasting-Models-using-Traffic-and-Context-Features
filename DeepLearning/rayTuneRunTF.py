from models.nbeatsx.nbeats_model import NBeatsX

from util.nbeatsx.ts_loader import TimeSeriesLoader
import shutup;shutup.please()
import os
import git
from datetime import datetime

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from models.StackedLSTMRegressor import StackedLSTMRegressor
from models.LSTMRegressor import LSTMRegressor
from models.nbeats import NBeatsNet

from util.DataLoadingUtil import create_dataset
from util.Evaluation import evaluate_model
from util import Config

import ray
from ray import air, tune
from ray.tune.schedulers import MedianStoppingRule
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.search.hyperopt import HyperOptSearch

import matplotlib
matplotlib.use('Agg')


def conduct_experiment(args):
    matplotlib.use('Agg')
    torch.set_float32_matmul_precision('high')
    os.environ['OMP_NUM_THREADS'] = '10'

    args['encode-segment-features'] = args['primary_features']
    args['encode-holidays'] = args['secondary_features']
    args['encode-severe-weather'] = args['secondary_features']
    args['encode-incidents'] = args['secondary_features']

    if args['model'] != 'NBEATSx':
        train_dataset, val_dataset, test_dataset = create_dataset(args)
        # create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=int(args['batch_size']), shuffle=True, pin_memory=True,
                                  num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=int(args['batch_size']), shuffle=False, pin_memory=True,
                                num_workers=4)
    else:
        train_dataset, val_dataset, test_dataset = create_dataset(args, drop_timestamp=False)
        train_loader = TimeSeriesLoader(dataset=train_dataset, input_size=args['window_size'],
                                        output_size=args['prediction_length'], batch_size=int(args['batch_size']),
                                        shuffle=True, num_workers=4)

        val_loader = TimeSeriesLoader(dataset=val_dataset, input_size=args['window_size'],
                                      output_size=args['prediction_length'], batch_size=int(args['batch_size']),
                                      shuffle=False, num_workers=4)

    # set random seed and create model
    pl.seed_everything(10)
    if args['model'] == 'LSTMRegressor':
        model_class = LSTMRegressor
        model = LSTMRegressor(window_size=args['window_size'], output_size=args['prediction_length'],
                              hidden_dim=int(args['hidden_dim']), n_layers=int(args['n_layers']),
                              input_size=args['feature_count'], scalers=args['scalers'],
                              learning_rate=args['learning_rate'], cell_type=args['cell_type'])
    elif args['model'] == 'StackedLSTMRegressor':
        model_class = StackedLSTMRegressor
        model = StackedLSTMRegressor(output_size=args['prediction_length'], input_size=args['feature_count'],
                                     hidden_dim_1=int(args['hidden_dim_1']), n_layers_1=int(args['n_layers_1']),
                                     hidden_dim_2=int(args['hidden_dim_2']), n_layers_2=int(args['n_layers_2']),
                                     scalers=args['scalers'], learning_rate=args['learning_rate'],
                                     cell_type=args['cell_type'])
    elif args['model'] == 'NBEATS':
        model_class = NBeatsNet
        args['thetas_dim'] = (int(args['thetas_dim']), int(args['thetas_dim']))
        model = NBeatsNet(forecast_length=args['prediction_length'], backcast_length=args['window_size'],
                          stack_types=args['stack_types'], nb_blocks_per_stack=int(args['nb_blocks_per_stack']),
                          thetas_dim=args['thetas_dim'], share_weights_in_stack=args['share_weights_in_stack'],
                          hidden_layer_units=int(args['hidden_layer_units']),
                          learning_rate=args['learning_rate'],
                          scalers=args['scalers'], device='cuda')
    elif args['model'] == 'NBEATS-trend-seasonality':
        model_class = NBeatsNet
        args['thetas_dim'] = (int(args['thetas_dim_trend']), int(args['thetas_dim_seasonality']))
        model = NBeatsNet(forecast_length=args['prediction_length'], backcast_length=args['window_size'],
                          stack_types=args['stack_types'], nb_blocks_per_stack=int(args['nb_blocks_per_stack']),
                          thetas_dim=args['thetas_dim'], share_weights_in_stack=args['share_weights_in_stack'],
                          hidden_layer_units=int(args['hidden_layer_units']),
                          learning_rate=args['learning_rate'],
                          scalers=args['scalers'], device='cuda')
    elif args['model'] == 'NBEATSx':
        model_class = NBeatsX
        n_hidden_1_list = [int(args['n_hidden_1']) for _ in range(0, int(args['n_layers_1']))]
        n_hidden_2_list = [int(args['n_hidden_2']) for _ in range(0, int(args['n_layers_2']))]
        args['exogenous_n_channels'] = max(args['exogenous_n_channels'], args['feature_count']-1)
        args['x_s_n_hidden'] = max(args['x_s_n_hidden'], args['feature_count']-1)
        model = NBeatsX(input_size_multiplier=args['window_size'] // args['prediction_length'],
                        output_size=args['prediction_length'], shared_weights=args['shared_weights'],
                        initialization='glorot_normal',
                        activation='relu',
                        stack_types=args['stack_types'],  # -> unclear
                        n_blocks=[int(args['n_blocks_1']), int(args['n_blocks_2'])],
                        n_layers=[int(args['n_layers_1']), int(args['n_layers_2'])],
                        n_hidden=[n_hidden_1_list, n_hidden_2_list],
                        n_harmonics=0,  # not used with exogenous_tcn
                        n_polynomials=0,  # not used with exogenous_tcn
                        x_s_n_hidden=int(args['x_s_n_hidden']),  # Number of encoded static features to calculate
                        exogenous_n_channels=int(args['exogenous_n_channels']),
                        t_cols=train_dataset.t_cols,
                        batch_normalization=True,
                        dropout_prob_theta=0.1,
                        dropout_prob_exogenous=0.1,
                        learning_rate=args['learning_rate'],
                        weight_decay=0,
                        l1_theta=0,
                        seasonality=24,  # only used with MASE loss
                        n_variables=train_loader.get_n_variables(),
                        scalers=args['scalers'])
    else:
        raise Exception(f'Unsupported model type {args["model"]}')

    # setup components for trainer
    logger = TensorBoardLogger(args['tensorboard_logger_location'], name="TensorboardLogs", version=".")  # logging results to a tensorboard
    # see docu: https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.EarlyStopping.html
    # Parameterization: https://stackoverflow.com/questions/43906048/which-parameters-should-be-used-for-early-stopping
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', min_delta=1e-4, patience=3)

    tune_report_cb = TuneReportCallback(
                {
                    "val_loss": "val_loss",
                    "val_mae": "val_mae",
                    "val_mse": "val_mse",
                    "val_rmse": "val_rmse",
                    "val_smape": "val_smape",
                },
                on="validation_end")


    accelerator = "cpu"
    if args['gpus'] == 1:
        accelerator = "gpu"

    trainer = pl.Trainer(max_epochs=args['max_epochs'], accelerator=accelerator,
                         callbacks=[tune_report_cb, early_stopping], logger=logger,
                         # make it silent:
                         enable_progress_bar=False, enable_model_summary=False)

    # start training
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Model evaluation
    model_path = trainer.checkpoint_callback.best_model_path
    evaluate_model(args, model_path, args['scalers'], test_dataset, model_class)
    metrics = trainer.logged_metrics

    # log validation metrics to tune and set status to done (done is not set automatically!)
    tune.report(val_loss=metrics['val_loss'].item(),
                val_mae=metrics['val_mae'].item(),
                val_mse=metrics['val_mse'].item(),
                val_rmse=metrics['val_rmse'].item(),
                val_smape=metrics['val_smape'].item(),
                done=True)


def get_simpleLSTM_config():
    # 4 parameters -> 40 - 80 trials
    model_config = {
        'batch_size': tune.quniform(32, 512, 32),
        'hidden_dim': tune.quniform(16, 112, 4),
        'n_layers': tune.quniform(4, 20, 2),
        'learning_rate': tune.loguniform(1e-4, 1e-1),
        'cell_type': 'LSTM',
        'model': 'LSTMRegressor',
    }

    return model_config, 40


def get_simpleGRU_config():
    # 4 parameters -> 40 - 80 trials
    model_config = {
        'batch_size': tune.quniform(32, 512, 32),
        'hidden_dim': tune.quniform(16, 112, 4),
        'n_layers': tune.quniform(4, 20, 2),
        'learning_rate': tune.loguniform(1e-4, 1e-1),
        'cell_type': 'GRU',
        'model': 'LSTMRegressor',
    }

    return model_config, 40


def get_stackedLSTM_config():
    # 6 parameters -> 60 - 120 trials
    model_config = {
        'batch_size': tune.quniform(32, 512, 32),
        'hidden_dim_1': tune.quniform(16, 112, 4),
        'n_layers_1': tune.quniform(4, 12, 2),
        'hidden_dim_2': tune.quniform(16, 112, 4),
        'n_layers_2': tune.quniform(4, 12, 2),
        'learning_rate': tune.loguniform(1e-4, 1e-1),
        'cell_type': 'LSTM',
        'model': 'StackedLSTMRegressor',
    }

    return model_config, 60


def get_stackedGRU_config():
    # 6 parameters -> 60 - 120 trials
    model_config = {
        'batch_size': tune.quniform(32, 512, 32),
        'hidden_dim_1': tune.quniform(16, 112, 4),
        'n_layers_1': tune.quniform(4, 12, 2),
        'hidden_dim_2': tune.quniform(16, 112, 4),
        'n_layers_2': tune.quniform(4, 12, 2),
        'learning_rate': tune.loguniform(1e-4, 1e-1),
        'cell_type': 'GRU',
        'model': 'StackedLSTMRegressor',
    }

    return model_config, 60


def get_nbeats_config():
    model_config = {
        'learning_rate': tune.loguniform(1e-4, 1e-1),
        'batch_size': tune.quniform(32, 512, 32),
        'stack_types': (NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK),
        'thetas_dim': tune.quniform(4, 50, 2),  # has to be of same length as stack types!
        'nb_blocks_per_stack': tune.quniform(1, 24, 1),
        'share_weights_in_stack': tune.choice([True, False]),  # is default
        'hidden_layer_units': tune.quniform(8, 256, 8),
        'model': 'NBEATS',
    }

    return model_config, 80


def get_nbeatsx_config():
    model_config = {
        'learning_rate': tune.loguniform(1e-4, 1e-1),
        'batch_size': tune.quniform(32, 1024, 32),
        'stack_types': ['exogenous_tcn', 'exogenous_tcn'],
        'n_blocks_1': tune.quniform(1, 2, 1),
        'n_layers_1': tune.quniform(1, 6, 1),
        'n_hidden_1': tune.quniform(32, 512, 32),
        'n_blocks_2': tune.quniform(1, 2, 1),
        'n_layers_2': tune.quniform(1, 6, 1),
        'n_hidden_2': tune.quniform(32, 512, 32),
        'exogenous_n_channels': tune.quniform(1, 32, 1),
        'x_s_n_hidden': tune.quniform(0, 32, 1),
        'shared_weights': tune.choice([True, False]),
        'model': 'NBEATSx',
    }

    return model_config, 130


def get_nbeats_config_trend_seasonality():
    model_config = {
        'learning_rate': tune.loguniform(1e-4, 1e-1),
        'batch_size': tune.quniform(32, 512, 32),
        'stack_types': (NBeatsNet.TREND_BLOCK, NBeatsNet.SEASONALITY_BLOCK),
        'thetas_dim_trend': tune.quniform(1, 4, 1),
        'thetas_dim_seasonality': tune.quniform(2, 128, 2),
        'nb_blocks_per_stack': tune.quniform(1, 24, 1),
        'share_weights_in_stack': tune.choice([True, False]),  # is default
        'hidden_layer_units': tune.quniform(8, 256, 8),
        'model': 'NBEATS-trend-seasonality',
    }

    return model_config, 90

def get_git_sha():
    try:
        repo = git.Repo(search_parent_directories=True)
        git_sha = repo.head.object.hexsha
    except git.exc.InvalidGitRepositoryError:
        # In case the directory is not used with git
        git_sha = "Not found"

    return git_sha


def select_model(model_name):
    if model_name == 'simpleLSTM':
        return get_simpleLSTM_config()
    elif model_name == 'simpleGRU':
        return get_simpleGRU_config()
    elif model_name == 'stackedLSTM':
        return get_stackedLSTM_config()
    elif model_name == 'stackedGRU':
        return get_stackedGRU_config()
    elif model_name == 'NBEATS':
        return get_nbeats_config()
    elif model_name == 'NBEATS-trend-seasonality':
        return get_nbeats_config_trend_seasonality()
    elif model_name == 'NBEATSx':
        return get_nbeatsx_config()


if __name__ == '__main__':
    os.environ['TUNE_MAX_PENDING_TRIALS_PG'] = '4'
    os.environ['RAY_memory_usage_threshold'] = '0.97'   # 0.95 is default!
    ray.init(include_dashboard=False, object_store_memory=int(4e9), local_mode=False)

    max_concurrent_trials = 4
    resources_per_trial = {"cpu": 12, "gpu": 0.19}

    model_config, model_trials = select_model(Config.model_name)

    exogenous_features = Config.model_name != 'NBEATS' and Config.model_name != 'NBEATS-trend-seasonality'
    # Number of experiments that should be conducted
    num_samples = 60 if exogenous_features else 0
    num_samples += model_trials

    config = {
        'tensorboard_logger_location': './',
        'dataset_path': Config.raw_dataset_path,
        'experiment_comment': '',
        'repository_version': get_git_sha(),
        'starting_date': datetime.today().strftime('%Y-%m-%d'),
        'resampling-parameter': "0.5H",
        'window_size': 48,
        'prediction_length': 8,
        'max_epochs': 100,
        'encode-time': True if exogenous_features else False,
        'primary_features': tune.choice([True, False]) if exogenous_features else False,     # 2 choices -> 30 trials
        'secondary_features': tune.choice([True, False]) if exogenous_features else False,   # 2 choices -> 30 trials
        'proportion-of-segments': 0.75,
        'gpus': 1,
        'number-days-training': 78,
        'number-days-validation': 21,
        'number-days-testing': 21
    }

    # merge config dictionaries
    config = {**config, **model_config}

    scheduler = MedianStoppingRule(
        time_attr='training_iteration',
        grace_period=5,
        min_samples_required=4,    # 3 is default
        min_time_slice=0,   # 0 is default value
        hard_stop=True      # True is default value
    )

    ray_tuner = tune.Tuner(
        tune.with_resources(trainable=tune.with_parameters(conduct_experiment), resources=resources_per_trial),
        tune_config=tune.TuneConfig(
            metric="val_loss", mode="min", scheduler=scheduler,
            num_samples=num_samples, max_concurrent_trials=max_concurrent_trials,
            search_alg=HyperOptSearch(random_state_seed=42),
        ),
        run_config=air.RunConfig(name=f'traffic_median_{Config.model_name}_4h_{config["resampling-parameter"]}', verbose=0),
        param_space=config,
    )

    result_grid = ray_tuner.fit()
    print("Best hyperparameters found were: ", result_grid.get_best_result().config)
    result_grid.get_dataframe().to_csv(f"results-{Config.model_name}-{config['resampling-parameter']}.csv", sep=";")
