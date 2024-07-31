import copy
import math
import os

from pytorch_lightning import LightningModule
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

from util import MongoDb, Config
from util.Metrics import smape
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util.nbeatsx.ts_loader import TimeSeriesLoader


def _nbeatsx_inference(batch, model):
    insample_y = model._to_tensor(batch['insample_y'])
    insample_x = model._to_tensor(batch['insample_x'])
    insample_mask = model._to_tensor(batch['insample_mask'])
    outsample_x = model._to_tensor(batch['outsample_x'])
    outsample_y = model._to_tensor(batch['outsample_y'])
    s_matrix = model._to_tensor(batch['s_matrix'])

    output = model(x_s=s_matrix, insample_y=insample_y, insample_x_t=insample_x, outsample_x_t=outsample_x,
                   insample_mask=insample_mask)
    targets = outsample_y
    return output, targets


def evaluate_instantiated_model(args, model, scalers, test_dataset, model_class, save_plots=False, model_path=None,
                                append_predicted_values=False, test_dataset_list=None):
    return _evaluate(model, args, scalers, test_dataset, model_class, save_plots=save_plots,
                     model_path=model_path, only_return_results=True, append_predicted_values=append_predicted_values,
                     test_dataset_list=test_dataset_list)


def evaluate_model(args, model_path, scalers, test_dataset, model_class, save_plots=False):
    loaded_model: LightningModule = model_class.load_from_checkpoint(model_path, scalers=scalers)
    return _evaluate(loaded_model, args, scalers, test_dataset, model_class, save_plots=save_plots,
                     model_path=model_path)


def _evaluate_dataloader_nbeatsX(test_loader, loaded_model, scalers):
    predicted_result, actual_result = [], []

    for i, data in enumerate(test_loader):
        result, targets = _nbeatsx_inference(data, loaded_model)
        result = result.to('cpu')

        actual_predicted_df = pd.DataFrame(
            data={"actual": targets.view(-1).tolist(), "predicted": result.view(-1).tolist()})
        inverse_transformed_values = scalers['speed_percentage'].inverse_transform(actual_predicted_df)
        actual_predicted_df["actual"] = inverse_transformed_values[:, [0]]
        actual_predicted_df["predicted"] = inverse_transformed_values[:, [1]]

        predicted_result.extend(actual_predicted_df["predicted"])
        actual_result.extend(actual_predicted_df["actual"])

    return predicted_result, actual_result


def _evaluate(loaded_model, args, scalers, test_dataset, model_class, save_plots=False, model_path=None,
              only_return_results=False, append_predicted_values=False, test_dataset_list=None):
    loaded_model.to(Config.device)
    loaded_model.eval()

    # batch_size = 1 if (save_plots or model_class.__qualname__ == 'NBeatsX') else 2048
    batch_size = 1 if save_plots else 2048
    evaluation_mode_nbeatsx = True if batch_size == 1 else False

    predicted_result, actual_result, predicted_plot, actual_plot = [], [], [], []
    edge_id = None
    counter = 0
    edge_id_list = []
    evaluation_samples = {}

    if save_plots:
        os.makedirs(args['save_location'], exist_ok=True)

    if model_class.__qualname__ == 'NBeatsX' and test_dataset_list is None:
        raise Exception("Not implemented. Provide datasets as list. Each dataset contains data of a single olr code.")
        # NBeatsX requires different dataset and dataloader object
        # test_loader = TimeSeriesLoader(dataset=test_dataset, input_size=args['window_size'], num_workers=1,
        #                                output_size=args['prediction_length'], batch_size=batch_size, shuffle=False,
        #                                evaluation_mode=evaluation_mode_nbeatsx)
    elif model_class.__qualname__ == 'NBeatsX' and type(test_dataset_list) == list:
        for dataset in test_dataset_list:
            test_loader = TimeSeriesLoader(dataset=dataset, input_size=args['window_size'], num_workers=1,
                                           output_size=args['prediction_length'], batch_size=batch_size, shuffle=False,
                                           evaluation_mode=True, skip_validation_leakage_overkill=True)

            new_prediction, new_truth = _evaluate_dataloader_nbeatsX(test_loader, loaded_model, scalers)
            actual_result += new_truth
            predicted_result += new_prediction

        resulting_metrics = {
            'pred_isfinite': True,
            'mae': mean_absolute_error(actual_result, predicted_result),
            'mse': mean_squared_error(actual_result, predicted_result),
            'rmse': math.sqrt(mean_squared_error(actual_result, predicted_result)),
            'smape': smape(np.array(actual_result), np.array(predicted_result)),
            'r^2': r2_score(actual_result, predicted_result),
            'mape': mean_absolute_percentage_error(actual_result, predicted_result)
        }

        if append_predicted_values:
            resulting_metrics['truth'] = actual_result
            resulting_metrics['predicted'] = predicted_result

        return resulting_metrics
    else:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=1)

    for i, data in enumerate(test_loader):
        if model_class.__qualname__ == 'NBeatsX':
            result, targets = _nbeatsx_inference(data, loaded_model)
        else:
            features, targets = data
            features = features.to(Config.device)
            result = loaded_model(features)

        if model_class.__qualname__ == 'NBeatsNet':
            _, forecast = result
            result = forecast.to('cpu')
        else:
            result = result.to('cpu')

        actual_predicted_df = pd.DataFrame(
            data={"actual": targets.view(-1).tolist(), "predicted": result.view(-1).tolist()})

        if scalers is not None and 'speed_percentage' in scalers:
            inverse_transformed_values = scalers['speed_percentage'].inverse_transform(actual_predicted_df)
            actual_predicted_df["actual"] = inverse_transformed_values[:, [0]]
            actual_predicted_df["predicted"] = inverse_transformed_values[:, [1]]

        predicted_result.extend(actual_predicted_df["predicted"])
        actual_result.extend(actual_predicted_df["actual"])

        if save_plots:
            if model_class.__qualname__ == 'NBeatsX':
                Exception('NBeatsX does not support this currently')

            # data for plots has to be reset between identifiers
            predicted_plot.extend(actual_predicted_df["predicted"])
            actual_plot.extend(actual_predicted_df["actual"])

            if edge_id is None:
                edge_id = test_dataset.get_identifier(i)
            elif edge_id != test_dataset.get_identifier(i) and counter < 5:
                evaluation_samples[str(counter)] = {
                    'edge_id': edge_id,
                    'actual': copy.deepcopy(actual_plot),
                    'predicted': copy.deepcopy(predicted_plot)
                }

                edge_id_list.append(edge_id)
                cm = 1 / 2.54
                plt.style.use('seaborn')
                plt.subplots(figsize=(40 * cm, 8 * cm))
                plt.plot(actual_plot, color='tab:blue', label='actual')
                plt.plot(predicted_plot, color='tab:orange', label='prediction')
                plt.legend()
                plt.savefig(f"{args['save_location']}/{counter}.jpg", dpi=300)
                plt.clf()  # clear the figure

                predicted_plot, actual_plot = [], []
                edge_id = test_dataset.get_identifier(i)
                counter += 1

    if np.isfinite(predicted_result).all():
        resulting_metrics = {
            'pred_isfinite': True,
            'mae': mean_absolute_error(actual_result, predicted_result),
            'mse': mean_squared_error(actual_result, predicted_result),
            'rmse': math.sqrt(mean_squared_error(actual_result, predicted_result)),
            'smape': smape(np.array(actual_result), np.array(predicted_result)),
            'r^2': r2_score(actual_result, predicted_result),
            'mape': mean_absolute_percentage_error(actual_result, predicted_result)
        }

        if append_predicted_values:
            resulting_metrics['truth'] = actual_result
            resulting_metrics['predicted'] = predicted_result
    else:
        resulting_metrics = {
            'pred_isfinite': False
        }

    if not only_return_results:
        if Config.save_to_mongodb:
            MongoDb.save_model_data(args, loaded_model, args['scalers'], resulting_metrics, evaluation_samples)

        if Config.save_to_files:
            with open(f"{args['save_location']}/out.txt", "x") as f:
                f.write(str(args))
                f.write("\n")
                if model_path:
                    f.write(str(model_path))
                    f.write("\n")
                f.write(str(edge_id_list))
                f.write("\n")
                f.write(str(resulting_metrics))

    return resulting_metrics
