from typing import Sequence, Any, Optional, Union, Tuple

import torch
from pandas import DataFrame
from torch import Tensor
import numpy as np
from torch_kalman.data_utils.utils import tens_to_long


class TimeSeriesBatch:
    """
    TimeSeriesBatch includes additional information about each of the Tensor's dimensions: the name for each group in the
    first dimension, the start datetime and datetime-unit for the second dimension, and the name of the measures for the
    third dimension.
    """

    def __init__(self,
                 tensor: Tensor,
                 group_names: Sequence[Any],
                 start_datetimes: Sequence[np.datetime64],
                 measures: Sequence[str],
                 time_unit: Optional[str] = None):

        if not isinstance(group_names, np.ndarray):
            group_names = np.array(group_names)

        if not isinstance(start_datetimes, np.ndarray):
            start_datetimes = np.array(start_datetimes)
            if time_unit is None:
                time_unit, _ = np.datetime_data(start_datetimes[0])
                if time_unit == 'ns':  # since it's the default, we can't assume this is the interval they wanted
                    raise RuntimeError("Please pass an `interval` argument to specify the datetime-interval.")
            start_datetimes = start_datetimes.astype(f"datetime64[{time_unit}]")

        if not isinstance(measures, np.ndarray):
            measures = np.array(measures)

        self.tensor = tensor
        self.group_names = group_names
        self.start_datetimes = start_datetimes
        self.measures = measures

    def subset(self, ind: Union[int, Sequence, slice]) -> 'TimeSeriesBatch':
        if isinstance(ind, int):
            ind = [ind]
        return self.__class__(self.tensor[ind], self.group_names[ind], self.start_datetimes[ind], self.measures)

    def subset_by_groups(self, groups: Sequence[Any]):
        group_idx = np.where(np.isin(self.group_names, groups))[0]
        return self.subset(group_idx)

    def with_new_tensor(self, tensor: Tensor) -> 'TimeSeriesBatch':
        """
        Create a new Batch with a different Tensor, but all other attributes the same.
        :param tensor:
        :return:
        """
        return self.__class__(tensor,
                              group_names=self.group_names,
                              start_datetimes=self.start_datetimes,
                              measures=self.measures)

    def datetimes(self) -> np.ndarray:
        return self.start_datetimes[:, None] + np.arange(0, self.tensor.shape[1])

    def split(self, split_frac: float) -> Tuple['TimeSeriesBatch', 'TimeSeriesBatch']:
        """
        Split data along a pre-post train/validation.
        """
        assert 0. < split_frac < 1.

        time_len = self.tensor.shape[1]
        idx = round(time_len * split_frac)
        if 1 < idx < time_len:
            train_batch = self.with_new_tensor(self.tensor[:, :idx, :])
            val_batch = self.__class__(self.tensor[:, idx:, :],
                                       self.group_names,
                                       self.start_datetimes + idx,
                                       self.measures)
        else:
            raise ValueError("`split_frac` too extreme, results in empty tensor.")
        return train_batch, val_batch

    def to_dataframe(self,
                     tensor: Optional[Union[Tensor, np.ndarray]] = None,
                     group_colname: str = 'group',
                     datetime_colname: str = 'datetime',
                     measure_colname: str = 'measure',
                     value_colname: str = 'value',
                     ) -> DataFrame:
        if tensor is None:
            tensor = self.tensor

        df = DataFrame(tens_to_long(tensor))

        # datetimes:
        time_unit, _ = np.datetime_data(self.start_datetimes[0])
        df[datetime_colname] = self.start_datetimes[df.dim0.values] + df.dim1.values.astype(f'timedelta64[{time_unit}]')

        # group:
        df[group_colname] = self.group_names[df.dim0.values]

        # measures:
        df[measure_colname] = self.measures[df.dim2.values]

        # value:
        df[value_colname] = df['value']

        return df.loc[:, [group_colname, datetime_colname, measure_colname, value_colname]].copy()

    @classmethod
    def from_dataframe(cls,
                       dataframe: DataFrame,
                       group_col: str,
                       datetime_col: str,
                       measure_cols: Sequence[str],
                       time_unit: str) -> 'TimeSeriesBatch':
        assert isinstance(group_col, str)
        assert isinstance(datetime_col, str)
        assert isinstance(measure_cols, (list, tuple))

        # sort by datetime:
        dataframe = dataframe.sort_values(datetime_col)

        # first pass for info:
        arrays, time_idxs, group_names, start_datetimes = [], [], [], []
        for g, df in dataframe.groupby(group_col):
            # group-names:
            group_names.append(g)

            # times:
            times = df[datetime_col].values
            min_time = times[0]
            start_datetimes.append(min_time)
            time_idx = (times - min_time).astype(f'timedelta64[{time_unit}]').view('int64')
            time_idxs.append(time_idx)

            # values:
            arrays.append(df.loc[:, measure_cols].values)

        # second pass organizes into tensor
        time_len = max(time_idx[-1] + 1 for time_idx in time_idxs)
        tens = torch.empty((len(arrays), time_len, len(measure_cols)))
        tens[:] = np.nan
        for i, (array, time_idx) in enumerate(zip(arrays, time_idxs)):
            tens[i, time_idx, :] = Tensor(array)

        return cls(tensor=tens,
                   group_names=group_names,
                   start_datetimes=start_datetimes,
                   measures=measure_cols,
                   time_unit=time_unit)
