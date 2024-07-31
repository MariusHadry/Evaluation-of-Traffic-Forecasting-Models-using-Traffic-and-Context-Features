import numpy as np
import torch as t
from util.nbeatsx.ts_dataset import TimeSeriesDataset
from torch.utils.data import DataLoader


class TimeSeriesLoader(DataLoader):
    def __init__(self, dataset: TimeSeriesDataset, input_size: int, batch_size: int, shuffle: bool,
                 output_size: int = 0, window_sampling_limit: int = np.Infinity, offset: int = 0, num_workers: int = 0,
                 evaluation_mode: bool=False, skip_validation_leakage_overkill=False):
        """
        Time Series Loader object, used to sample time series from TimeSeriesDataset object.
        Parameters
        ----------
        dataset: TimeSeriesDataset
        Time Series Dataet object which contains data in PyTorch tensors optimized for sampling.
        offset: int
            Equivalent to timestamps in test (data in test will not be sampled). It is used to filter
            the PyTorch tensor containing the time series, to avoid using the future during training.
        window_sampling_limit: int
            Equivalent to calibration window. Length of the history (prior to offset) which will be sampled
        input_size: int
            Size of inputs of each window (only for NBEATS), eg. 7 days
        ouput_size: int
            Forecasting horizon
        batch_size: int
            Number of batches (windows) to sample
        is_train_loader: bool
            True: will only sample time stamps with 1s in mask, False: will only sample time stamps with 0s in mask
        shuffle: bool
            Indicates if windows should be shuffled. True is used for training and False for predicting.
        evaluation_mode: bool
            if set to true, it is assumed that only a single olr code is in the input data!
        """
        # Dataloader attributes
        super().__init__(dataset, batch_size, shuffle, num_workers=num_workers)
        self.window_sampling_limit = window_sampling_limit
        self.input_size = input_size
        self.output_size = output_size
        self.offset = offset
        self.ts_dataset = dataset
        self.t_cols = self.ts_dataset.t_cols
        self.shuffle = shuffle  # Boolean to shuffle data, useful for validation
        self.evaluation_mode = evaluation_mode
        self.skip_validation_leakage_overkill = skip_validation_leakage_overkill

        # Create rolling window matrix in advanced for faster access to data and broadcasted s_matrix
        self._create_train_data()

    def _update_sampling_windows_idxs(self):
        if self.evaluation_mode:
            # This might only work if there is a single olr code in the data. Check what happens if there are multiple olr codes!
            ts_windows: t.Tensor = self.ts_windows
            tensor_size = ts_windows.size(dim=0)
            tmp = list(np.arange(0, tensor_size))
            return tmp

        # Only sample during training windows with at least one active output mask and input mask
        outsample_condition = t.sum(self.ts_windows[:, self.t_cols.index('outsample_mask'), -self.output_size:], axis=1)
        insample_condition = t.sum(self.ts_windows[:, self.t_cols.index('insample_mask'), :self.input_size], axis=1)
        sampling_idx = t.nonzero(outsample_condition * insample_condition > 0)  # element-wise product
        sampling_idx = list(sampling_idx.flatten().numpy())
        return sampling_idx

    def _create_windows_tensor(self):
        # Memory efficiency is gained from keeping across dataloaders common ts_tensor in dataset
        # Filter function is used to define train tensor and validation tensor with the offset
        # Default ts_idxs=ts_idxs sends all the data
        tensor, right_padding, train_mask = self.ts_dataset.get_filtered_ts_tensor(offset=self.offset,
                                                                                   output_size=self.output_size,
                                                                                   window_sampling_limit=self.window_sampling_limit)
        tensor = t.Tensor(tensor)

        # Outsample mask checks existance of values in ts, train_mask mask is used to filter out validation
        # is_train_loader inverts the train_mask in case the dataloader is in validation mode
        #   -> inverting the train mask is not required as this parameter is not used!
        mask = train_mask
        tensor[:, self.t_cols.index('outsample_mask'), :] = tensor[:, self.t_cols.index('outsample_mask'), :] * mask

        # remove padding
        # padder = t.nn.ConstantPad1d(padding=(self.input_size, right_padding), value=0)
        padder = t.nn.ConstantPad1d(padding=(0, 0), value=0)
        tensor = padder(tensor)

        # Last output_size outsample_mask and y to 0
        if not self.skip_validation_leakage_overkill:
            tensor[:, self.t_cols.index('speed_percentage'), -self.output_size:] = 0  # overkill to ensure no validation leakage
        tensor[:, self.t_cols.index('outsample_mask'), -self.output_size:] = 0

        # Creating rolling windows and 'flattens' them
        windows = tensor.unfold(dimension=-1, size=self.input_size + self.output_size, step=1)
        # n_serie, n_channel, n_time, window_size -> n_serie, n_time, n_channel, window_size
        # print(f'n_serie, n_channel, n_time, window_size = {windows.shape}')
        windows = windows.permute(0, 2, 1, 3)
        # print(f'n_serie, n_time, n_channel, window_size = {windows.shape}')
        windows = windows.reshape(-1, self.ts_dataset.n_channels, self.input_size + self.output_size)

        # Broadcast s_matrix: This works because unfold in windows_tensor, orders: time, serie
        s_matrix = self.ts_dataset.s_matrix.repeat(repeats=int(len(windows) / self.ts_dataset.n_series), axis=0)

        return windows, s_matrix

    def __len__(self):
        # return number of batches within this dataloader
        return int(np.ceil(len(self.windows_sampling_idx) / self.batch_size))  # Must be multiple of batch_size for paralel gpu

    def __iter__(self):
        if self.shuffle:
            sample_idxs = np.random.choice(a=self.windows_sampling_idx,
                                           size=len(self.windows_sampling_idx), replace=False)
        else:
            sample_idxs = self.windows_sampling_idx

        assert len(sample_idxs) > 0, 'Check the data as sample_idxs are empty'

        n_batches = int(np.ceil(len(sample_idxs) / self.batch_size))  # Must be multiple of batch_size for paralel gpu

        for idx in range(n_batches):
            ws_idxs = sample_idxs[(idx * self.batch_size): (idx + 1) * self.batch_size]
            batch = self.__get_item__(index=ws_idxs)
            yield batch

    def __get_item__(self, index):
        return self._nbeats_batch(index)

    def _nbeats_batch(self, index):
        # Access precomputed rolling window matrix (RAM intensive)
        windows = self.ts_windows[index]
        s_matrix = self.s_matrix[index]

        insample_y = windows[:, self.t_cols.index('speed_percentage'), :self.input_size]
        insample_x = windows[:, (self.t_cols.index('speed_percentage') + 1):self.t_cols.index('insample_mask'), :self.input_size]
        insample_mask = windows[:, self.t_cols.index('insample_mask'), :self.input_size]

        outsample_y = windows[:, self.t_cols.index('speed_percentage'), self.input_size:]
        outsample_x = windows[:, (self.t_cols.index('speed_percentage') + 1):self.t_cols.index('insample_mask'), self.input_size:]
        outsample_mask = windows[:, self.t_cols.index('outsample_mask'), self.input_size:]

        batch = {'s_matrix': s_matrix,
                 'insample_y': insample_y, 'insample_x': insample_x, 'insample_mask': insample_mask,
                 'outsample_y': outsample_y, 'outsample_x': outsample_x, 'outsample_mask': outsample_mask}
        return batch

    def _create_train_data(self):
        """
        """
        # Create rolling window matrix for fast information retrieval
        self.ts_windows, self.s_matrix = self._create_windows_tensor()
        self.n_windows = len(self.ts_windows)
        self.windows_sampling_idx = self._update_sampling_windows_idxs()

    def update_offset(self, offset):
        if offset == self.offset:
            return  # Avoid extra computation
        self.offset = offset
        self._create_train_data()

    def get_meta_data_col(self, col):
        return self.ts_dataset.get_meta_data_col(col)

    def get_n_variables(self):
        return self.ts_dataset.n_x, self.ts_dataset.n_s

    def get_n_series(self):
        return self.ts_dataset.n_series

    def get_max_len(self):
        return self.ts_dataset.max_len

    def get_n_channels(self):
        return self.ts_dataset.n_channels

    def get_X_cols(self):
        return self.ts_dataset.X_cols

    def get_frequency(self):
        return self.ts_dataset.frequency
