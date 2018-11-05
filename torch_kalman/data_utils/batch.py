from typing import Sequence, Any, Optional, Union, Tuple

import torch
import pandas as pd
from torch import Tensor
import numpy as np


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

    def subset_by_groups(self, groups: Sequence[Any]) -> 'TimeSeriesBatch':
        """
        Get the subset of the batch corresponding to groups. Note that the ordering in the output will match the original
        ordering (not that of `group`), and that duplicates will be dropped.
        """
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
                     datetime_colname: str = 'datetime'
                     ) -> pd.DataFrame:
        if tensor is None:
            tensor = self.tensor
        else:
            assert tensor.shape == self.tensor.shape, f"The `tensor` has wrong shape, expected `{self.tensor.shape}`."

        dfs = []
        for g, group_name in enumerate(self.group_names):
            # get values, don't store trailing nans:
            values = tensor[g, :, :].detach().numpy()
            all_nan_per_row = np.min(np.isnan(values), axis=1)
            end_idx = np.max(np.where(~all_nan_per_row)[0]) + 1
            # convert to dataframe:
            df = pd.DataFrame(data=values[:end_idx, :], columns=self.measures)
            df[group_colname] = group_name
            df[datetime_colname] = self.start_datetimes[g] + np.arange(0, len(df.index))
            dfs.append(df)

        return pd.concat(dfs)

    @classmethod
    def from_dataframe(cls,
                       dataframe: pd.DataFrame,
                       group_colname: str,
                       datetime_colname: str,
                       measure_colnames: Sequence[str],
                       time_unit: str,
                       missing: Optional[float] = None,
                       ) -> 'TimeSeriesBatch':
        assert isinstance(group_colname, str)
        assert isinstance(datetime_colname, str)
        assert isinstance(measure_colnames, (list, tuple))
        assert len(measure_colnames) == len(set(measure_colnames))

        # sort by datetime:
        dataframe = dataframe.sort_values(datetime_colname)

        # first pass for info:
        arrays, time_idxs, group_names, start_datetimes = [], [], [], []
        for g, df in dataframe.groupby(group_colname):
            # group-names:
            group_names.append(g)

            # times:
            times = df[datetime_colname].values
            min_time = times[0]
            start_datetimes.append(min_time)
            time_idx = (times - min_time).astype(f'timedelta64[{time_unit}]').view('int64')
            time_idxs.append(time_idx)

            # values:
            arrays.append(df.loc[:, measure_colnames].values)

        # second pass organizes into tensor
        time_len = max(time_idx[-1] + 1 for time_idx in time_idxs)
        tens = torch.empty((len(arrays), time_len, len(measure_colnames)))
        tens[:] = np.nan
        for i, (array, time_idx) in enumerate(zip(arrays, time_idxs)):
            tens[i, time_idx, :] = Tensor(array)

        if missing is not None:
            tens[torch.isnan(tens)] = missing

        return cls(tensor=tens,
                   group_names=group_names,
                   start_datetimes=start_datetimes,
                   measures=measure_colnames,
                   time_unit=time_unit)
