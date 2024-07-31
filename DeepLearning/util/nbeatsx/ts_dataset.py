import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from collections import defaultdict


class TimeSeriesDataset(Dataset):
    def __init__(self,
                 Y_df: pd.DataFrame,
                 X_df: pd.DataFrame = None,
                 S_df: pd.DataFrame = None,
                 f_cols: list = None):
        """
        Time Series Dataset object.
        Parameters
        ----------
        Y_df: DataFrame
            DataFrame with target variable. Must contain columns ['olr_code', 'ds', 'speed_percentage']
        X_df: DataFrame
            DataFrame with temporal exogenous variables. Must contain columns ['olr_code', 'timestamp']
        S_df: DataFrame
            DataFrame with static exogenous variables. Must contain columns ['olr_code', 'timestamp']
        f_cols: list
            Name of columns which future exogenous variables (eg. forecasts)
        """
        assert type(Y_df) == pd.core.frame.DataFrame
        assert all([(col in Y_df) for col in ['olr_code', 'timestamp', 'speed_percentage']])
        if X_df is not None:
            assert type(X_df) == pd.core.frame.DataFrame
            assert all([(col in X_df) for col in ['olr_code', 'timestamp']])

        # Pandas dataframes to data lists
        ts_data, s_data, self.meta_data, self.t_cols, self.X_cols = self._df_to_lists(Y_df=Y_df, S_df=S_df, X_df=X_df)

        # Dataset attributes
        self.n_series = len(ts_data)
        self.max_len = max([len(ts['speed_percentage']) for ts in ts_data])
        self.n_channels = len(self.t_cols)  # y, X_cols, insample_mask and outsample_mask
        self.frequency = pd.infer_freq(Y_df.head()['timestamp'])
        self.f_cols = f_cols

        # Number of X and S features
        self.n_x = 0 if X_df is None else len(self.X_cols)
        self.n_s = 0 if S_df is None else S_df.shape[1] - 1  # -1 for olr_code

        # Balances panel and creates
        # numpy  s_matrix of shape (n_series, n_s)
        # numpy ts_tensor of shape (n_series, n_channels, max_len) n_channels = y + X_cols + masks
        self.ts_tensor, self.s_matrix, self.len_series = self._create_tensor(ts_data, s_data)

        # train mask is not used in the adapted code format. Thus, all data is used within the
        ts_train_mask = np.ones(self.max_len)
        assert len(ts_train_mask) == self.max_len, f'Outsample mask must have {self.max_len} length'

        self._declare_outsample_train_mask(ts_train_mask)

    def __len__(self):
        return len(self.len_series)

    def _df_to_lists(self, Y_df, S_df, X_df):
        """
        """
        unique_ids = Y_df['olr_code'].unique()

        if X_df is not None:
            X_cols = [col for col in X_df.columns if col not in ['olr_code', 'timestamp']]
        else:
            X_cols = []

        if S_df is not None:
            S_cols = [col for col in S_df.columns if col not in ['olr_code']]
        else:
            S_cols = []

        ts_data = []
        s_data = []
        meta_data = []
        for i, u_id in enumerate(unique_ids):
            top_row = Y_df['olr_code'].searchsorted(u_id, 'left').item()
            bottom_row = Y_df['olr_code'].searchsorted(u_id, 'right').item()
            serie = Y_df[top_row:bottom_row]['speed_percentage'].values
            last_ds_i = Y_df[top_row:bottom_row]['timestamp'].max()

            # Y values
            ts_data_i = {'speed_percentage': serie}

            # X values
            for X_col in X_cols:
                serie = X_df[top_row:bottom_row][X_col].values
                ts_data_i[X_col] = serie
            ts_data.append(ts_data_i)

            # S values
            s_data_i = defaultdict(list)
            for S_col in S_cols:
                s_data_i[S_col] = S_df.loc[S_df['olr_code'] == u_id, S_col].values
            s_data.append(s_data_i)

            # Metadata
            meta_data_i = {'olr_code': u_id,
                           'last_ds': last_ds_i}
            meta_data.append(meta_data_i)

        t_cols = ['speed_percentage'] + X_cols + ['insample_mask', 'outsample_mask']

        return ts_data, s_data, meta_data, t_cols, X_cols

    def _create_tensor(self, ts_data, s_data):
        """
        s_matrix of shape (n_series, n_s)
        ts_tensor of shape (n_series, n_channels, max_len) n_channels = y + X_cols + masks
        """
        s_matrix = np.zeros((self.n_series, self.n_s))
        ts_tensor = np.zeros((self.n_series, self.n_channels, self.max_len))

        len_series = []
        for idx in range(self.n_series):
            ts_idx = np.array(list(ts_data[idx].values()))

            ts_tensor[idx, :self.t_cols.index('insample_mask'), -ts_idx.shape[1]:] = ts_idx
            ts_tensor[idx, self.t_cols.index('insample_mask'), -ts_idx.shape[1]:] = 1

            # To avoid sampling windows without inputs available to predict we shift -1
            # outsample_mask will be completed with the train_mask, this ensures available data
            ts_tensor[idx, self.t_cols.index('outsample_mask'), -(ts_idx.shape[1]):] = 1
            s_matrix[idx, :] = list(s_data[idx].values())
            len_series.append(ts_idx.shape[1])

        return ts_tensor, s_matrix, np.array(len_series)

    def _declare_outsample_train_mask(self, ts_train_mask):
        # Update attribute and ts_tensor
        self.ts_train_mask = ts_train_mask

    def get_meta_data_col(self, col):
        """
        """
        col_values = [x[col] for x in self.meta_data]
        return col_values

    def get_filtered_ts_tensor(self, offset, output_size, window_sampling_limit, ts_idxs=None):
        last_outsample_ds = self.max_len - offset + output_size
        first_ds = max(last_outsample_ds - window_sampling_limit - output_size, 0)
        if ts_idxs is None:
            filtered_ts_tensor = self.ts_tensor[:, :, first_ds:last_outsample_ds]
        else:
            filtered_ts_tensor = self.ts_tensor[ts_idxs, :, first_ds:last_outsample_ds]
        right_padding = max(last_outsample_ds - self.max_len,
                            0)  # To padd with zeros if there is "nothing" to the right
        ts_train_mask = self.ts_train_mask[first_ds:last_outsample_ds]

        assert np.sum(np.isnan(filtered_ts_tensor)) < 1.0, \
            f'The balanced balanced filtered_tensor has {np.sum(np.isnan(filtered_ts_tensor))} nan values'
        return filtered_ts_tensor, right_padding, ts_train_mask

    def get_f_idxs(self, cols):
        # Check if cols are available f_cols and return the idxs
        assert all(col in self.f_cols for col in cols), f'Some variables in {cols} are not available in f_cols.'
        f_idxs = [self.X_cols.index(col) for col in cols]
        return f_idxs
