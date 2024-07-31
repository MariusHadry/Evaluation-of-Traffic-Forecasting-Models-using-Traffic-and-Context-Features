import torchmetrics
import pandas as pd
import torch
from torch import nn

from models.LSTMRegressor import LSTMRegressor

pd.options.mode.chained_assignment = None  # default='warn'


class StackedLSTMRegressor(LSTMRegressor):
    def __init__(self, output_size, hidden_dim_1, hidden_dim_2, n_layers_1, n_layers_2, scalers,
                 input_size=1, learning_rate=0.001, cell_type='LSTM'):
        """
        Stacking multiple LSTM layers with different hidden dimensions with a linear layer to determine the final
        forecast.

        :param output_size: number of future steps to be forecast
        :param hidden_dim_1: hidden dimension of first LSTM
        :param hidden_dim_2: hidden dimension of second LSTM
        :param n_layers_1: layers of first LSTM
        :param n_layers_2: layers of second LSTM
        :param window_size: number of previous time steps used for forecasting
        :param scalers: scalers for calculating metrics
        :param input_size: Number of features in the input
        :param learning_rate: learning rate for the model, defaults to 0.001
        """

        super(LSTMRegressor, self).__init__()
        self.save_hyperparameters(ignore=['_scalers', 'scalers'])

        self.learning_rate = learning_rate

        self._scalers = scalers
        self.hidden_dim_1 = hidden_dim_1
        self.n_layers_1 = n_layers_1
        self.hidden_dim_2 = hidden_dim_2
        self.n_layers_2 = n_layers_2

        self.cell_type = cell_type

        if self.cell_type == 'LSTM':
            self.cell_1 = nn.LSTM(input_size, self.hidden_dim_1, self.n_layers_1, bidirectional=False, batch_first=True)
            self.cell_2 = nn.LSTM(self.hidden_dim_1, self.hidden_dim_2, self.n_layers_2, bidirectional=False,
                                  batch_first=True)
        elif self.cell_type == 'GRU':
            self.cell_1 = nn.GRU(input_size, self.hidden_dim_1, self.n_layers_1, bidirectional=False, batch_first=True)
            self.cell_2 = nn.GRU(self.hidden_dim_1, self.hidden_dim_2, self.n_layers_2, bidirectional=False,
                                 batch_first=True)
        else:
            Exception(f"unspecified cell type {self.cell_type}")

        self.layernorm_1 = nn.LayerNorm(self.hidden_dim_1)
        self.fc = nn.Linear(hidden_dim_2 * self.n_layers_1, output_size)

        self.loss = nn.MSELoss()
        self._metrics_mae = torchmetrics.MeanAbsoluteError().to(self.device)
        self._metrics_mse = torchmetrics.MeanSquaredError().to(self.device)
        self._metrics_smape = torchmetrics.SymmetricMeanAbsolutePercentageError().to(self.device)

        self.image_counter = 0
        self.previous_epoch = -1

    def forward(self, x):
        h_1 = None

        if self.cell_type == 'LSTM':
            out, (h_1, c_1) = self.cell_1(x)
        elif self.cell_type == 'GRU':
            out, h_1 = self.cell_1(x)
        else:
            Exception(f"unspecified cell type {self.cell_type}")

        # permute hidden tensor to fulfill batch_first
        h_1 = torch.permute(h_1, (1, 0, 2))
        h_1 = self.layernorm_1(h_1)
        out, _ = self.cell_2(h_1)

        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out
