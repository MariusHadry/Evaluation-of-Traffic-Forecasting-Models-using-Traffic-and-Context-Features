import math
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from typing import Tuple

import torchmetrics
from functools import partial
import pytorch_lightning as pl

from models.nbeatsx.tcn import TemporalConvNet
from matplotlib import pyplot as plt


def filter_input_vars(insample_y, insample_x_t, outsample_x_t, t_cols, include_var_dict):
    # This function is specific for the EPF task
    if torch.cuda.is_available():
        device = insample_x_t.get_device()
    else:
        device = 'cpu'
    outsample_y = torch.zeros((insample_y.shape[0], 1, outsample_x_t.shape[2])).to(device)

    insample_y_aux = torch.unsqueeze(insample_y, dim=1)

    insample_x_t_aux = torch.cat([insample_y_aux, insample_x_t], dim=1)
    outsample_x_t_aux = torch.cat([outsample_y, outsample_x_t], dim=1)
    x_t = torch.cat([insample_x_t_aux, outsample_x_t_aux], dim=-1)
    batch_size, n_channels, input_size = x_t.shape

    assert input_size == 168 + 24, f'input_size {input_size} not 168+24'

    x_t = x_t.reshape(batch_size, n_channels, 8, 24)

    input_vars = []
    for var in include_var_dict.keys():
        if len(include_var_dict[var]) > 0:
            t_col_idx = t_cols.index(var)
            t_col_filter = include_var_dict[var]
            if var != 'week_day':
                input_vars += [x_t[:, t_col_idx, t_col_filter, :]]
            else:
                assert t_col_filter == [-1], f'Day of week must be of outsample not {t_col_filter}'
                day_var = x_t[:, t_col_idx, t_col_filter, [0]]
                day_var = day_var.view(batch_size, -1)

    x_t_filter = torch.cat(input_vars, dim=1)
    x_t_filter = x_t_filter.view(batch_size, -1)

    if len(include_var_dict['week_day']) > 0:
        x_t_filter = torch.cat([x_t_filter, day_var], dim=1)

    return x_t_filter


class _StaticFeaturesEncoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(_StaticFeaturesEncoder, self).__init__()
        layers = [nn.Dropout(p=0.5),
                  nn.Linear(in_features=in_features, out_features=out_features),
                  nn.ReLU()]
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        return x


class NBeatsBlock(nn.Module):
    """
    N-BEATS block which takes a basis function as an argument.
    """

    def __init__(self, x_t_n_inputs: int, x_s_n_inputs: int, x_s_n_hidden: int, theta_n_dim: int, basis: nn.Module,
                 n_layers: int, theta_n_hidden: list, include_var_dict, t_cols, batch_normalization: bool,
                 dropout_prob: float, activation: str):
        super().__init__()

        if x_s_n_inputs == 0:
            x_s_n_hidden = 0
        theta_n_hidden = [x_t_n_inputs + x_s_n_hidden] + theta_n_hidden

        self.x_s_n_inputs = x_s_n_inputs
        self.x_s_n_hidden = x_s_n_hidden
        self.include_var_dict = include_var_dict
        self.t_cols = t_cols
        self.batch_normalization = batch_normalization
        self.dropout_prob = dropout_prob
        self.activations = {'relu': nn.ReLU(),
                            'softplus': nn.Softplus(),
                            'tanh': nn.Tanh(),
                            'selu': nn.SELU(),
                            'lrelu': nn.LeakyReLU(),
                            'prelu': nn.PReLU(),
                            'sigmoid': nn.Sigmoid()}

        hidden_layers = []
        for i in range(n_layers):

            # Batch norm after activation
            hidden_layers.append(nn.Linear(in_features=theta_n_hidden[i], out_features=theta_n_hidden[i + 1]))
            hidden_layers.append(self.activations[activation])

            if self.batch_normalization:
                hidden_layers.append(nn.BatchNorm1d(num_features=theta_n_hidden[i + 1]))

            if self.dropout_prob > 0:
                hidden_layers.append(nn.Dropout(p=self.dropout_prob))

        output_layer = [nn.Linear(in_features=theta_n_hidden[-1], out_features=theta_n_dim)]
        layers = hidden_layers + output_layer

        # x_s_n_inputs is computed with data, x_s_n_hidden is provided by user, if 0 no statics are used
        if (self.x_s_n_inputs > 0) and (self.x_s_n_hidden > 0):
            self.static_encoder = _StaticFeaturesEncoder(in_features=x_s_n_inputs, out_features=x_s_n_hidden)
        self.layers = nn.Sequential(*layers)
        self.basis = basis

    def forward(self, insample_y: torch.Tensor, insample_x_t: torch.Tensor,
                outsample_x_t: torch.Tensor, x_s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.include_var_dict is not None:
            insample_y = filter_input_vars(insample_y=insample_y, insample_x_t=insample_x_t,
                                           outsample_x_t=outsample_x_t,
                                           t_cols=self.t_cols, include_var_dict=self.include_var_dict)

        # Static exogenous
        if (self.x_s_n_inputs > 0) and (self.x_s_n_hidden > 0):
            x_s = self.static_encoder(x_s)
            insample_y = torch.cat((insample_y, x_s), 1)

        # Compute local projection weights and projection
        theta = self.layers(insample_y)
        backcast, forecast = self.basis(theta, insample_x_t, outsample_x_t)

        return backcast, forecast


class NBeatsX(pl.LightningModule):
    """
    N-Beats-X Model.
    """

    def __init__(self,
                 n_variables,
                 input_size_multiplier,
                 output_size,
                 shared_weights,
                 activation,
                 initialization,
                 stack_types,
                 n_blocks,
                 n_layers,
                 n_hidden,
                 n_harmonics,
                 n_polynomials,
                 exogenous_n_channels,
                 t_cols,
                 batch_normalization,
                 dropout_prob_theta,
                 dropout_prob_exogenous,
                 x_s_n_hidden,
                 learning_rate,
                 weight_decay,
                 l1_theta,
                 seasonality,
                 scalers,
                 device='cuda'):
        """

        :param n_variables: result from dataloader.get_n_variables()
        :param input_size_multiplier: int
            Multiplier to get insample size.
            Insample size = input_size_multiplier * output_size
        :param output_size: int
            Forecast horizon.
        :param shared_weights: bool
            If True, repeats first block.
        :param activation: str
            Activation function.
            An item from ['relu', 'softplus', 'tanh', 'selu', 'lrelu', 'prelu', 'sigmoid'].
        :param initialization: str
            Initialization function.
            An item from ['orthogonal', 'he_uniform', 'glorot_uniform', 'glorot_normal', 'lecun_normal'].
        :param stack_types: List[str]
            List of stack types.
            Subset from ['seasonality', 'trend', 'identity', 'exogenous', 'exogenous_tcn', 'exogenous_wavenet'].
        :param n_blocks: List[int]
            Number of blocks for each stack type.
            Note that len(n_blocks) = len(stack_types).
        :param n_layers: List[int]
            Number of layers for each stack type.
            Note that len(n_layers) = len(stack_types).
        :param n_hidden: List[List[int]]
            Structure of hidden layers for each stack type.
            Each internal list should contain the number of units of each hidden layer.
            Note that len(n_hidden) = len(stack_types).
        :param n_harmonics: List[int]
            Number of harmonic terms for each stack type.
            Note that len(n_harmonics) = len(stack_types).
        :param n_polynomials: List[int]
            Number of polynomial terms for each stack type.
            Note that len(n_polynomials) = len(stack_types).
        :param exogenous_n_channels:
            Exogenous channels for non-interpretable exogenous basis.
        :param t_cols: List
            Ordered list of ['y'] + X_cols + ['available_mask', 'sample_mask'].
            Can be taken from the dataset.
        :param batch_normalization: bool
            Whether perform batch normalization.
        :param dropout_prob_theta: float
            Float between (0, 1).
            Dropout for Nbeats basis.
        :param dropout_prob_exogenous: float
            Float between (0, 1).
            Dropout for exogenous basis.
        :param x_s_n_hidden: int
            Number of encoded static features to calculate.
        :param learning_rate: float
            Learning rate between (0, 1).
        :param weight_decay: float
            L2 penalty for optimizer.
        :param l1_theta: float
            L1 regularization for the loss function.
        :param seasonality: int
            Time series seasonality.
            Usually 7 for daily data, 12 for monthly data and 4 for weekly data.
        """
        super(NBeatsX, self).__init__()
        self.save_hyperparameters(ignore=['_scalers', 'scalers'])
        self._scalers = scalers
        self._device = torch.device(device)

        # ------------------------ Model Attributes ------------------------#
        # Architecture parameters
        self.input_size = int(input_size_multiplier * output_size)
        self.output_size = output_size
        self.shared_weights = shared_weights
        self.activation = activation
        self.initialization = initialization
        self.stack_types = stack_types
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_harmonics = n_harmonics
        self.n_polynomials = n_polynomials
        self.exogenous_n_channels = exogenous_n_channels

        # Regularization and optimization parameters
        self.batch_normalization = batch_normalization
        self.dropout_prob_theta = dropout_prob_theta
        self.dropout_prob_exogenous = dropout_prob_exogenous
        self.x_s_n_hidden = x_s_n_hidden
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.l1_theta = l1_theta
        self.l1_conv = 1e-3  # Not a hyperparameter

        # Data parameters
        self.seasonality = seasonality
        self.include_var_dict = None
        self.t_cols = t_cols

        self.n_x_t, self.n_x_s = n_variables
        self.blocks: nn.ModuleList = torch.nn.ModuleList(self._create_stack())


        self.loss = nn.MSELoss()
        self._metrics_mae = torchmetrics.MeanAbsoluteError().to(self.device)
        self._metrics_mse = torchmetrics.MeanSquaredError().to(self.device)
        self._metrics_smape = torchmetrics.SymmetricMeanAbsolutePercentageError().to(self.device)

        self.image_counter = 0
        self.previous_epoch = -1

    def _init_weights(self, module, initialization):
        if type(module) == torch.nn.Linear:
            if initialization == 'orthogonal':
                torch.nn.init.orthogonal_(module.weight)
            elif initialization == 'he_uniform':
                torch.nn.init.kaiming_uniform_(module.weight)
            elif initialization == 'he_normal':
                torch.nn.init.kaiming_normal_(module.weight)
            elif initialization == 'glorot_uniform':
                torch.nn.init.xavier_uniform_(module.weight)
            elif initialization == 'glorot_normal':
                torch.nn.init.xavier_normal_(module.weight)
            elif initialization == 'lecun_normal':
                pass
            else:
                assert 1 < 0, f'Initialization {initialization} not found'

    def _create_stack(self):
        if self.include_var_dict is not None:
            x_t_n_inputs = self.output_size * int(sum([len(x) for x in self.include_var_dict.values()]))

            # Correction because week_day only adds 1 no output_size
            if len(self.include_var_dict['week_day']) > 0:
                x_t_n_inputs = x_t_n_inputs - self.output_size + 1
        else:
            x_t_n_inputs = self.input_size

        # ------------------------ Model Definition ------------------------#
        block_list = []
        self.blocks_regularizer = []
        for i in range(len(self.stack_types)):
            for block_id in range(self.n_blocks[i]):

                # Batch norm only on first block
                if (len(block_list) == 0) and (self.batch_normalization):
                    batch_normalization_block = True
                else:
                    batch_normalization_block = False

                # Dummy of regularizer in block. Override with 1 if exogenous_block
                self.blocks_regularizer += [0]

                # Shared weights
                if self.shared_weights and block_id > 0:
                    nbeats_block = block_list[-1]
                else:
                    if self.stack_types[i] == 'seasonality':
                        nbeats_block = NBeatsBlock(x_t_n_inputs=x_t_n_inputs,
                                                   x_s_n_inputs=self.n_x_s,
                                                   x_s_n_hidden=self.x_s_n_hidden,
                                                   theta_n_dim=4 * int(
                                                       np.ceil(self.n_harmonics / 2 * self.output_size) - (
                                                                   self.n_harmonics - 1)),
                                                   basis=SeasonalityBasis(harmonics=self.n_harmonics,
                                                                          backcast_size=self.input_size,
                                                                          forecast_size=self.output_size),
                                                   n_layers=self.n_layers[i],
                                                   theta_n_hidden=self.n_hidden[i],
                                                   include_var_dict=self.include_var_dict,
                                                   t_cols=self.t_cols,
                                                   batch_normalization=batch_normalization_block,
                                                   dropout_prob=self.dropout_prob_theta,
                                                   activation=self.activation)
                    elif self.stack_types[i] == 'trend':
                        nbeats_block = NBeatsBlock(x_t_n_inputs=x_t_n_inputs,
                                                   x_s_n_inputs=self.n_x_s,
                                                   x_s_n_hidden=self.x_s_n_hidden,
                                                   theta_n_dim=2 * (self.n_polynomials + 1),
                                                   basis=TrendBasis(degree_of_polynomial=self.n_polynomials,
                                                                    backcast_size=self.input_size,
                                                                    forecast_size=self.output_size),
                                                   n_layers=self.n_layers[i],
                                                   theta_n_hidden=self.n_hidden[i],
                                                   include_var_dict=self.include_var_dict,
                                                   t_cols=self.t_cols,
                                                   batch_normalization=batch_normalization_block,
                                                   dropout_prob=self.dropout_prob_theta,
                                                   activation=self.activation)
                    elif self.stack_types[i] == 'identity':
                        nbeats_block = NBeatsBlock(x_t_n_inputs=x_t_n_inputs,
                                                   x_s_n_inputs=self.n_x_s,
                                                   x_s_n_hidden=self.x_s_n_hidden,
                                                   theta_n_dim=self.input_size + self.output_size,
                                                   basis=IdentityBasis(backcast_size=self.input_size,
                                                                       forecast_size=self.output_size),
                                                   n_layers=self.n_layers[i],
                                                   theta_n_hidden=self.n_hidden[i],
                                                   include_var_dict=self.include_var_dict,
                                                   t_cols=self.t_cols,
                                                   batch_normalization=batch_normalization_block,
                                                   dropout_prob=self.dropout_prob_theta,
                                                   activation=self.activation)
                    elif self.stack_types[i] == 'exogenous':
                        nbeats_block = NBeatsBlock(x_t_n_inputs=x_t_n_inputs,
                                                   x_s_n_inputs=self.n_x_s,
                                                   x_s_n_hidden=self.x_s_n_hidden,
                                                   theta_n_dim=2 * self.n_x_t,
                                                   basis=ExogenousBasisInterpretable(),
                                                   n_layers=self.n_layers[i],
                                                   theta_n_hidden=self.n_hidden[i],
                                                   include_var_dict=self.include_var_dict,
                                                   t_cols=self.t_cols,
                                                   batch_normalization=batch_normalization_block,
                                                   dropout_prob=self.dropout_prob_theta,
                                                   activation=self.activation)
                    elif self.stack_types[i] == 'exogenous_tcn':
                        nbeats_block = NBeatsBlock(x_t_n_inputs=x_t_n_inputs,
                                                   x_s_n_inputs=self.n_x_s,
                                                   x_s_n_hidden=self.x_s_n_hidden,
                                                   theta_n_dim=2 * (self.exogenous_n_channels),
                                                   basis=ExogenousBasisTCN(self.exogenous_n_channels, self.n_x_t),
                                                   n_layers=self.n_layers[i],
                                                   theta_n_hidden=self.n_hidden[i],
                                                   include_var_dict=self.include_var_dict,
                                                   t_cols=self.t_cols,
                                                   batch_normalization=batch_normalization_block,
                                                   dropout_prob=self.dropout_prob_theta,
                                                   activation=self.activation)
                    elif self.stack_types[i] == 'exogenous_wavenet':
                        nbeats_block = NBeatsBlock(x_t_n_inputs=x_t_n_inputs,
                                                   x_s_n_inputs=self.n_x_s,
                                                   x_s_n_hidden=self.x_s_n_hidden,
                                                   theta_n_dim=2 * (self.exogenous_n_channels),
                                                   basis=ExogenousBasisWavenet(self.exogenous_n_channels, self.n_x_t),
                                                   n_layers=self.n_layers[i],
                                                   theta_n_hidden=self.n_hidden[i],
                                                   include_var_dict=self.include_var_dict,
                                                   t_cols=self.t_cols,
                                                   batch_normalization=batch_normalization_block,
                                                   dropout_prob=self.dropout_prob_theta,
                                                   activation=self.activation)
                        self.blocks_regularizer[-1] = 1
                    else:
                        assert 1 < 0, f'Block type not found!'
                # Select type of evaluation and apply it to all layers of block
                init_function = partial(self._init_weights, initialization=self.initialization)
                nbeats_block.layers.apply(init_function)
                block_list.append(nbeats_block)
        return block_list

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(x, dtype=torch.float32).to(self.device)
        return tensor

    def forward(self, insample_y: torch.Tensor, insample_x_t: torch.Tensor, insample_mask: torch.Tensor,
                outsample_x_t: torch.Tensor, x_s: torch.Tensor, return_decomposition=False):

        residuals = insample_y.flip(dims=(-1,))
        insample_x_t = insample_x_t.flip(dims=(-1,))
        insample_mask = insample_mask.flip(dims=(-1,))

        forecast = insample_y[:, -1:]  # Level with Naive1
        block_forecasts = []
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(insample_y=residuals, insample_x_t=insample_x_t,
                                             outsample_x_t=outsample_x_t, x_s=x_s)
            residuals = (residuals - backcast) * insample_mask
            forecast = forecast + block_forecast
            block_forecasts.append(block_forecast)

        # (n_batch, n_blocks, n_time)
        block_forecasts = torch.stack(block_forecasts)
        block_forecasts = block_forecasts.permute(1, 0, 2)

        if return_decomposition:
            return forecast, block_forecasts
        else:
            return forecast

    def decomposed_prediction(self, insample_y: torch.Tensor, insample_x_t: torch.Tensor, insample_mask: torch.Tensor,
                              outsample_x_t: torch.Tensor):

        residuals = insample_y.flip(dims=(-1,))
        insample_x_t = insample_x_t.flip(dims=(-1,))
        insample_mask = insample_mask.flip(dims=(-1,))

        forecast = insample_y[:, -1:]  # Level with Naive1
        forecast_components = []
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(residuals, insample_x_t, outsample_x_t)
            residuals = (residuals - backcast) * insample_mask
            forecast = forecast + block_forecast
            forecast_components.append(block_forecast)
        return forecast, forecast_components

    def configure_optimizers(self):
        params = self.parameters()
        optimizer = optim.Adam(params=params, lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        insample_y = self._to_tensor(train_batch['insample_y'])
        insample_x = self._to_tensor(train_batch['insample_x'])
        insample_mask = self._to_tensor(train_batch['insample_mask'])
        outsample_x = self._to_tensor(train_batch['outsample_x'])
        outsample_y = self._to_tensor(train_batch['outsample_y'])
        outsample_mask = self._to_tensor(train_batch['outsample_mask'])
        s_matrix = self._to_tensor(train_batch['s_matrix'])

        output = self(x_s=s_matrix, insample_y=insample_y, insample_x_t=insample_x, outsample_x_t=outsample_x,
                      insample_mask=insample_mask)

        targets = outsample_y
        loss = self.loss(output, targets)
        self.log('train_loss', loss, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, val_batch, batch_idx):
        insample_y = self._to_tensor(val_batch['insample_y'])
        insample_x = self._to_tensor(val_batch['insample_x'])
        insample_mask = self._to_tensor(val_batch['insample_mask'])
        outsample_x = self._to_tensor(val_batch['outsample_x'])
        outsample_y = self._to_tensor(val_batch['outsample_y'])
        outsample_mask = self._to_tensor(val_batch['outsample_mask'])
        s_matrix = self._to_tensor(val_batch['s_matrix'])

        output = self(x_s=s_matrix, insample_y=insample_y, insample_x_t=insample_x, outsample_x_t=outsample_x,
                      insample_mask=insample_mask)

        targets = outsample_y
        loss = self.loss(output, targets)


        inv_trans_output = torch.from_numpy(
            self._scalers['speed_percentage'].inverse_transform(output.cpu().reshape(-1, 1)))
        inv_trans_target = torch.from_numpy(
            self._scalers['speed_percentage'].inverse_transform(targets.cpu().reshape(-1, 1)))
        # inv_trans_output = output.cpu().reshape(-1, 1)
        # inv_trans_target = targets.cpu().reshape(-1, 1)

        self._metrics_mae.update(inv_trans_output, inv_trans_target)
        self._metrics_mse.update(inv_trans_output, inv_trans_target)
        self._metrics_smape.update(inv_trans_output, inv_trans_target)

        self.log('val_loss', loss, prog_bar=True)

        if self.previous_epoch != self.current_epoch and self.image_counter < 3:
            plt.style.use('seaborn')
            plt.plot(inv_trans_target, color='tab:blue', label='actual')
            plt.plot(inv_trans_output, color='tab:orange', label='prediction')
            plt.legend()
            self.logger.experiment.add_figure(f"epoch={self.current_epoch};sample={self.image_counter}", plt.gcf())
            self.image_counter += 1

        if self.image_counter == 3:
            self.previous_epoch = self.current_epoch
            self.image_counter = 0

    def predict(self, batch):
        insample_y = self._to_tensor(batch['insample_y'])
        insample_x = self._to_tensor(batch['insample_x'])
        insample_mask = self._to_tensor(batch['insample_mask'])
        outsample_x = self._to_tensor(batch['outsample_x'])
        outsample_y = self._to_tensor(batch['outsample_y'])
        outsample_mask = self._to_tensor(batch['outsample_mask'])
        s_matrix = self._to_tensor(batch['s_matrix'])

        model_output = self(x_s=s_matrix, insample_y=insample_y, insample_x_t=insample_x, outsample_x_t=outsample_x,
                      insample_mask=insample_mask)

        target = outsample_y

        return model_output, target

    def on_validation_epoch_end(self) -> None:
        self.log('val_mae', self._metrics_mae.compute(), prog_bar=False)
        self.log('val_mse', self._metrics_mse.compute(), prog_bar=False)
        self.log('val_rmse', math.sqrt(self._metrics_mse.compute().to('cpu')), prog_bar=False)
        self.log('val_smape', self._metrics_smape.compute(), prog_bar=False)
        self._metrics_mae = torchmetrics.MeanAbsoluteError().to(self.device)
        self._metrics_mse = torchmetrics.MeanSquaredError().to(self.device)
        self._metrics_smape = torchmetrics.SymmetricMeanAbsolutePercentageError().to(self.device)


class IdentityBasis(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size

    def forward(self, theta: torch.Tensor, insample_x_t: torch.Tensor, outsample_x_t: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        backcast = theta[:, :self.backcast_size]
        forecast = theta[:, -self.forecast_size:]
        return backcast, forecast


class TrendBasis(nn.Module):
    def __init__(self, degree_of_polynomial: int, backcast_size: int, forecast_size: int):
        super().__init__()
        polynomial_size = degree_of_polynomial + 1
        self.backcast_basis = nn.Parameter(
            torch.tensor(np.concatenate([np.power(np.arange(backcast_size, dtype=np.float) / backcast_size, i)[None, :]
                                         for i in range(polynomial_size)]), dtype=torch.float32), requires_grad=False)
        self.forecast_basis = nn.Parameter(
            torch.tensor(np.concatenate([np.power(np.arange(forecast_size, dtype=np.float) / forecast_size, i)[None, :]
                                         for i in range(polynomial_size)]), dtype=torch.float32), requires_grad=False)

    def forward(self, theta: torch.Tensor, insample_x_t: torch.Tensor, outsample_x_t: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        cut_point = self.forecast_basis.shape[0]
        backcast = torch.einsum('bp,pt->bt', theta[:, cut_point:], self.backcast_basis)
        forecast = torch.einsum('bp,pt->bt', theta[:, :cut_point], self.forecast_basis)
        return backcast, forecast


class SeasonalityBasis(nn.Module):
    def __init__(self, harmonics: int, backcast_size: int, forecast_size: int):
        super().__init__()
        frequency = np.append(np.zeros(1, dtype=np.float32),
                              np.arange(harmonics, harmonics / 2 * forecast_size,
                                        dtype=np.float32) / harmonics)[None, :]
        backcast_grid = -2 * np.pi * (
                np.arange(backcast_size, dtype=np.float32)[:, None] / forecast_size) * frequency
        forecast_grid = 2 * np.pi * (
                np.arange(forecast_size, dtype=np.float32)[:, None] / forecast_size) * frequency

        backcast_cos_template = torch.tensor(np.transpose(np.cos(backcast_grid)), dtype=torch.float32)
        backcast_sin_template = torch.tensor(np.transpose(np.sin(backcast_grid)), dtype=torch.float32)
        backcast_template = torch.cat([backcast_cos_template, backcast_sin_template], dim=0)

        forecast_cos_template = torch.tensor(np.transpose(np.cos(forecast_grid)), dtype=torch.float32)
        forecast_sin_template = torch.tensor(np.transpose(np.sin(forecast_grid)), dtype=torch.float32)
        forecast_template = torch.cat([forecast_cos_template, forecast_sin_template], dim=0)

        self.backcast_basis = nn.Parameter(backcast_template, requires_grad=False)
        self.forecast_basis = nn.Parameter(forecast_template, requires_grad=False)

    def forward(self, theta: torch.Tensor, insample_x_t: torch.Tensor, outsample_x_t: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        cut_point = self.forecast_basis.shape[0]
        backcast = torch.einsum('bp,pt->bt', theta[:, cut_point:], self.backcast_basis)
        forecast = torch.einsum('bp,pt->bt', theta[:, :cut_point], self.forecast_basis)
        return backcast, forecast


class ExogenousBasisInterpretable(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, theta: torch.Tensor, insample_x_t: torch.Tensor, outsample_x_t: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        backcast_basis = insample_x_t
        forecast_basis = outsample_x_t

        cut_point = forecast_basis.shape[1]
        backcast = torch.einsum('bp,bpt->bt', theta[:, cut_point:], backcast_basis)
        forecast = torch.einsum('bp,bpt->bt', theta[:, :cut_point], forecast_basis)
        return backcast, forecast


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class ExogenousBasisWavenet(nn.Module):
    def __init__(self, out_features, in_features, num_levels=4, kernel_size=3, dropout_prob=0):
        super().__init__()
        # Shape of (1, in_features, 1) to broadcast over b and t
        self.weight = nn.Parameter(torch.Tensor(1, in_features, 1), requires_grad=True)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(0.5))

        padding = (kernel_size - 1) * (2 ** 0)
        input_layer = [nn.Conv1d(in_channels=in_features, out_channels=out_features,
                                 kernel_size=kernel_size, padding=padding, dilation=2 ** 0),
                       Chomp1d(padding),
                       nn.ReLU(),
                       nn.Dropout(dropout_prob)]
        conv_layers = []
        for i in range(1, num_levels):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            conv_layers.append(nn.Conv1d(in_channels=out_features, out_channels=out_features,
                                         padding=padding, kernel_size=3, dilation=dilation))
            conv_layers.append(Chomp1d(padding))
            conv_layers.append(nn.ReLU())
        conv_layers = input_layer + conv_layers

        self.wavenet = nn.Sequential(*conv_layers)

    def transform(self, insample_x_t, outsample_x_t):
        input_size = insample_x_t.shape[2]

        x_t = torch.cat([insample_x_t, outsample_x_t], dim=2)

        x_t = x_t * self.weight  # Element-wise multiplication, broadcasted on b and t. Weights used in L1 regularization
        x_t = self.wavenet(x_t)[:]

        backcast_basis = x_t[:, :, :input_size]
        forecast_basis = x_t[:, :, input_size:]

        return backcast_basis, forecast_basis

    def forward(self, theta: torch.Tensor, insample_x_t: torch.Tensor, outsample_x_t: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        backcast_basis, forecast_basis = self.transform(insample_x_t, outsample_x_t)

        cut_point = forecast_basis.shape[1]
        backcast = torch.einsum('bp,bpt->bt', theta[:, cut_point:], backcast_basis)
        forecast = torch.einsum('bp,bpt->bt', theta[:, :cut_point], forecast_basis)
        return backcast, forecast


class ExogenousBasisTCN(nn.Module):
    def __init__(self, out_features, in_features, num_levels=4, kernel_size=2, dropout_prob=0):
        super().__init__()
        n_channels = num_levels * [out_features]
        self.tcn = TemporalConvNet(num_inputs=in_features, num_channels=n_channels, kernel_size=kernel_size,
                                   dropout=dropout_prob)

    def transform(self, insample_x_t, outsample_x_t):
        input_size = insample_x_t.shape[2]

        x_t = torch.cat([insample_x_t, outsample_x_t], dim=2)

        x_t = self.tcn(x_t)[:]
        backcast_basis = x_t[:, :, :input_size]
        forecast_basis = x_t[:, :, input_size:]

        return backcast_basis, forecast_basis

    def forward(self, theta: torch.Tensor, insample_x_t: torch.Tensor, outsample_x_t: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        backcast_basis, forecast_basis = self.transform(insample_x_t, outsample_x_t)

        cut_point = forecast_basis.shape[1]
        backcast = torch.einsum('bp,bpt->bt', theta[:, cut_point:], backcast_basis)
        forecast = torch.einsum('bp,bpt->bt', theta[:, :cut_point], forecast_basis)
        return backcast, forecast
