import pandas as pd
from torch.utils.data import Dataset
import numpy as np


class TrafficDataset(Dataset):
    def __init__(self, data, target, window_size, prediction_length, ts_identifier_col='olr_code', drop_time=True):
        super().__init__()
        self._len = 0
        self._window = window_size
        self._prediction_length = prediction_length
        count_lookup_df = data[ts_identifier_col].value_counts()

        index_list = []
        identifier_list = []

        for olr_code, occurrences in count_lookup_df.items():
            current_data_chunk = data[data[ts_identifier_col] == olr_code]
            offset = current_data_chunk.index[0]
            current_data_chunk.insert(0, 'running_id', range(0, len(current_data_chunk)))
            time_diff = current_data_chunk['timestamp'].diff()
            gaps = time_diff[time_diff > pd.Timedelta(hours=1)]

            if len(gaps) > 0:
                for i in gaps.index:
                    print(f'{i}, {olr_code}')
                    if (i - offset) > (self._window + self._prediction_length):
                        # add to index list until offset, adjust offset so that we are after gap?
                        len_current_series = max(0, i - offset - self._prediction_length - self._window)
                        tmp = np.arange(offset, offset + len_current_series)
                        index_list.extend(tmp)
                        # occurrences left after gap!
                        occurrences = occurrences - len(tmp) - self._prediction_length - self._window

                    # skip until after gap
                    offset = i

                # check if there is anything to add after gaps!
                #   +2 because we want to include the last index and the offset index!
                len_current_series = max(0, current_data_chunk.index[-1] - offset - self._prediction_length - self._window + 2)
                tmp = np.arange(offset, offset + len_current_series)
                index_list.extend(tmp)

                amount_to_fill = len(index_list) - len(identifier_list)
                identifier_list.extend(np.full(amount_to_fill, olr_code))
            else:
                # check if there is anything to add after gaps!
                len_current_series = max(0, occurrences - self._prediction_length - self._window + 1)
                tmp = np.arange(offset, offset + len_current_series)
                index_list.extend(tmp)
                identifier_list.extend(np.full(len_current_series, olr_code))

        to_drop = [ts_identifier_col]
        if drop_time:
            to_drop += ["timestamp"]
        data = data.drop(to_drop, axis=1)
        self._data = data.astype(np.float32)
        self._target = target.astype(np.float32)
        self._index_list = index_list
        self._segment_identifier_list = identifier_list

    def __getitem__(self, item):
        tmp_idx = self._index_list[item]
        x = self._data.iloc[tmp_idx:tmp_idx+self._window, :].values
        y = self._target.iloc[tmp_idx+self._window:tmp_idx+self._window+self._prediction_length].values
        return x, y

    def __len__(self):
        return len(self._index_list)

    def get_identifier(self, item):
        # should return the identifier for a given item so that one can reconstruct which time series the sample
        # was drawn from
        return self._segment_identifier_list[item]


class TrafficDatasetNoSkip(Dataset):
    """
    TrafficDataset Class that does not skip gaps in the data abut assumes interpolation/imputation of underlying data.
    """

    def __init__(self, data, target, window_size, prediction_length, ts_identifier_col='olr_code'):
        super().__init__()
        self._len = 0
        self._window = window_size
        self._prediction_length = prediction_length
        count_lookup_df = data[ts_identifier_col].value_counts()

        # determines the start of the current time series, assumes that the data is ordered by time and
        # timeseries identifier
        offset = 0

        index_list = []
        identifier_list = []

        for olr_code, occurrences in count_lookup_df.items():
            len_current_series = max(0, occurrences - self._prediction_length - self._window + 1)

            tmp = np.arange(offset, offset + len_current_series)
            index_list.append(tmp)
            identifier_list.append(np.full(len_current_series, olr_code))

            offset += occurrences

        data = data.drop([ts_identifier_col], axis=1)
        self._data = data.astype(np.float32)
        self._target = target.astype(np.float32)
        self._index_df = np.concatenate(index_list)
        self._identifier = np.concatenate(identifier_list)

    def __getitem__(self, item):
        tmp_idx = self._index_df[item]
        x = self._data.iloc[tmp_idx:tmp_idx+self._window, :].values
        y = self._target.iloc[tmp_idx+self._window:tmp_idx+self._window+self._prediction_length].values
        return x, y

    def __len__(self):
        return len(self._index_df)

    def get_identifier(self, item):
        # should return the identifier for a given item so that one can reconstruct which time series the sample
        # was drawn from
        return self._identifier[item]
