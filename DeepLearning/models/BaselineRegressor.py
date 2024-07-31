
import torch
import pandas as pd
import pytorch_lightning as pl

pd.options.mode.chained_assignment = None  # default='warn'


class BaselineRegressor(pl.LightningModule):
    def __init__(self, output_size):
        """
        Uses the last known value to forecast the output_size. Simply replicates the last known value for that task.
        Currently only works for batch_size = 1

        input_size: Number of features in the input
        output_size: number of items to be outputted
        """
        super(BaselineRegressor, self).__init__()
        self._output_size = output_size

    def forward(self, x):
        # TODO: currently only works for batch size = 1 (which suffices for evaluation purposes)
        # batch_size = x.size(0)
        out = x[0, -1, 0]
        out = torch.full((1, self._output_size), out)
        return out
