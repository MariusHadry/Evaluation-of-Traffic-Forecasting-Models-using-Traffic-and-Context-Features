import shutup;shutup.please()
import math

import torchmetrics
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
import pytorch_lightning as pl

pd.options.mode.chained_assignment = None  # default='warn'


class LSTMRegressor(pl.LightningModule):
    def __init__(self, output_size, hidden_dim, n_layers, window_size, scalers, input_size=1, learning_rate=0.001,
                 cell_type='LSTM'):
        """
        LSTM Regressor using a layer of LSTMs and a linear layer to make the final forecast.

        :param output_size: number of future steps to be forecast
        :param hidden_dim: hidden state dimension for the LSTM
        :param n_layers: number of layers to stack for the LSTM
        :param window_size: number of previous time steps used for forecasting
        :param scalers: scalers for inverting target scaling when calculating metrics
        :param input_size: Number of features used for the forecast
        :param learning_rate: learning rate for
        """

        super(LSTMRegressor, self).__init__()
        self.save_hyperparameters(ignore=['_scalers', 'scalers'])

        self.learning_rate = learning_rate

        self._scalers = scalers
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        if cell_type == 'LSTM':
            self.cell_1 = nn.LSTM(input_size, hidden_dim, n_layers, bidirectional=False, batch_first=True)
        elif cell_type == 'GRU':
            self.cell_1 = nn.GRU(input_size, hidden_dim, n_layers, bidirectional=False, batch_first=True)
        else:
            Exception(f"unspecified cell type {cell_type}")
        self.fc = nn.Linear(hidden_dim * window_size, output_size)

        self.loss = nn.MSELoss()
        self._metrics_mae = torchmetrics.MeanAbsoluteError().to(self.device)
        self._metrics_mse = torchmetrics.MeanSquaredError().to(self.device)
        self._metrics_smape = torchmetrics.SymmetricMeanAbsolutePercentageError().to(self.device)

        self.image_counter = 0
        self.previous_epoch = -1

    def forward(self, x):
        out, _ = self.cell_1(x)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out

    def get_hidden(self, batch_size, n_layers, hidden_dim):
        hidden_state = torch.zeros(n_layers, batch_size, hidden_dim, device=self.device)
        cell_state = torch.zeros(n_layers, batch_size, hidden_dim, device=self.device)
        return hidden_state, cell_state

    def configure_optimizers(self):
        params = self.parameters()
        optimizer = optim.Adam(params=params, lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        features, targets = train_batch
        output = self(features)
        output = output.view(-1)
        targets = targets.view(-1)
        loss = self.loss(output, targets)
        self.log('train_loss', loss, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, val_batch, batch_idx):
        features, targets = val_batch
        output = self(features)
        output = output.view(-1)
        targets = targets.view(-1)
        loss = self.loss(output, targets)

        if self._scalers is not None and 'speed_percentage' in self._scalers:
            output = torch.from_numpy(self._scalers['speed_percentage'].inverse_transform(output.cpu().reshape(-1, 1)))
            targets = torch.from_numpy(self._scalers['speed_percentage'].inverse_transform(targets.cpu().reshape(-1, 1)))

        self._metrics_mae.update(output, targets)
        self._metrics_mse.update(output, targets)
        self._metrics_smape.update(output, targets)

        self.log('val_loss', loss, prog_bar=True)

        if self.previous_epoch != self.current_epoch and self.image_counter < 3:
            plt.style.use('seaborn')
            plt.plot(targets.cpu(), color='tab:blue', label='actual')
            plt.plot(output.cpu(), color='tab:orange', label='prediction')
            plt.legend()
            self.logger.experiment.add_figure(f"epoch_{self.current_epoch}_sample_{self.image_counter}", plt.gcf())
            self.image_counter += 1

        if self.image_counter == 3:
            self.previous_epoch = self.current_epoch
            self.image_counter = 0

    def on_validation_epoch_end(self) -> None:
        self.log('val_mae', self._metrics_mae.compute(), prog_bar=False)
        self.log('val_mse', self._metrics_mse.compute(), prog_bar=False)
        self.log('val_rmse', math.sqrt(self._metrics_mse.compute().to('cpu')), prog_bar=False)
        self.log('val_smape', self._metrics_smape.compute(), prog_bar=False)
        self._metrics_mae = torchmetrics.MeanAbsoluteError().to(self.device)
        self._metrics_mse = torchmetrics.MeanSquaredError().to(self.device)
        self._metrics_smape = torchmetrics.SymmetricMeanAbsolutePercentageError().to(self.device)
