import itertools
from typing import Sequence, Any, Union, Optional, Tuple, Iterable
from warnings import warn

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import TensorDataset

from torch_kalman.internals.repr import NiceRepr


class TimeSeriesDataset(NiceRepr, TensorDataset):
    """
    TimeSeriesDataset includes additional information about each of the Tensors' dimensions: the name for each group in
    the first dimension, the start (date)time (and optionally datetime-unit) for the second dimension, and the name of
    the measures for the third dimension.

    Note that unlike TensorDataset, indexing a TimeSeriesDataset returns another TimeSeriesDataset, not a tuple of
    tensors. So when using TimeSeriesDataset, use `DataLoader(collate_fn=TimeSeriesDataset.collate)`.
    """
    supported_dt_units = {'Y', 'D', 'h', 'm', 's'}
    _repr_attrs = ('sizes', 'measures')

    def __init__(self,
                 *tensors: Tensor,
                 group_names: Sequence[Any],
                 start_times: Union[np.ndarray, Sequence],
                 measures: Sequence[Sequence[str]],
                 dt_unit: Optional[str]):

        if not isinstance(group_names, np.ndarray):
            group_names = np.array(group_names)

        assert len(group_names) == len(start_times)
        assert len(tensors) == len(measures)

        for i, (tensor, tensor_measures) in enumerate(zip(tensors, measures)):
            if len(tensor.shape) < 3:
                raise ValueError(f"Tensor {i} has < 3 dimensions")
            if tensor.shape[0] != len(group_names):
                raise ValueError(f"Tensor {i}'s first dimension has length != {len(group_names)}.")
            if tensor.shape[2] != len(tensor_measures):
                raise ValueError(f"Tensor {i}'s 3rd dimension has length != len({tensor_measures}).")

        self.measures = tuple(tuple(m) for m in measures)
        self.all_measures = tuple(itertools.chain.from_iterable(self.measures))
        self.group_names = group_names
        self.start_times = self._validate_start_times(start_times, dt_unit)
        self.dt_unit = dt_unit
        super().__init__(*tensors)

    def times(self) -> np.ndarray:
        num_timesteps = max(tensor.shape[1] for tensor in self.tensors)
        offset = np.arange(0, num_timesteps)
        if self.dt_unit == 'W':
            offset *= 7
        return self.start_times[:, None] + offset

    def datetimes(self) -> np.ndarray:
        return self.times()

    @property
    def start_datetimes(self) -> np.ndarray:
        return self.start_times

    @property
    def sizes(self) -> Sequence:
        return [t.size() for t in self.tensors]

    # Subsetting ------------------------:
    def __getitem__(self, item: Union[int, Sequence, slice]) -> 'TimeSeriesDataset':
        if isinstance(item, int):
            item = [item]
        return type(self)(
            *super(TimeSeriesDataset, self).__getitem__(item),
            group_names=self.group_names[item],
            start_times=self.start_times[item],
            measures=self.measures,
            dt_unit=self.dt_unit
        )

    def get_groups(self, groups: Sequence[Any]) -> 'TimeSeriesDataset':
        """
        Get the subset of the batch corresponding to groups. Note that the ordering in the output will match the original
        ordering (not that of `group`), and that duplicates will be dropped.
        """
        group_idx = np.where(np.isin(self.group_names, groups))[0]
        return self[group_idx]

    def split_times(self, split_frac: float) -> Tuple['TimeSeriesDataset', 'TimeSeriesDataset']:
        """
        Split data along a pre-post (typically: train/validation).
        """
        assert 0. < split_frac < 1.

        num_timesteps = max(tensor.shape[1] for tensor in self.tensors)
        idx = round(num_timesteps * split_frac)
        if 1 < idx < num_timesteps:
            train_batch = self.with_new_tensors(*(tensor[:, :idx, :] for tensor in self.tensors))
            val_batch = type(self)(*(tensor[:, idx:, :] for tensor in self.tensors),
                                   self.group_names,
                                   self.times()[:, idx],
                                   self.measures,
                                   dt_unit=self.dt_unit)
        else:
            raise ValueError("`split_frac` too extreme, results in empty tensor.")
        return train_batch, val_batch

    def split_measures(self, *measure_groups) -> 'TimeSeriesDataset':
        """
        :param measure_groups: Each argument should be an indexer (i.e. list of ints or a slice), or should be a list
        of measure-names.
        :return: A TimeSeriesDataset, now with multiple tensors for the measure-groups
        """
        if len(self.measures) > 1:
            raise RuntimeError(f"Can only split measures if there's only one group, but instead:\n{self.measures}")
        self_tensor = self.tensors[0]
        self_measures = self.measures[0]

        idxs = []
        for measure_group in measure_groups:
            if isinstance(measure_group, slice) or isinstance(measure_group[0], int):
                idxs.append(measure_group)
            else:
                idxs.append([self_measures.index(m) for m in measure_group])

        self_measures = np.array(self_measures)
        return type(self)(
            *(self_tensor[:, :, idx] for idx in idxs),
            start_times=self.start_times,
            group_names=self.group_names,
            measures=[tuple(self_measures[idx]) for idx in idxs],
            dt_unit=self.dt_unit
        )

    # Creation/Transformation ------------------------:
    @classmethod
    def collate(cls, batch: Sequence['TimeSeriesDataset']) -> 'TimeSeriesDataset':

        to_concat = {
            'tensors': [batch[0].tensors],
            'group_names': [batch[0].group_names],
            'start_times': [batch[0].start_times]
        }
        fixed = {'dt_unit': batch[0].dt_unit, 'measures': batch[0].measures}
        for i, ts_dataset in enumerate(batch[1:], 1):
            for attr, appendlist in to_concat.items():
                to_concat[attr].append(getattr(ts_dataset, attr))
            for attr, required_val in fixed.items():
                new_val = getattr(ts_dataset, attr)
                if new_val != required_val:
                    raise ValueError(f"Element {i} has `{attr}` = {new_val}, but for element 0 it's {required_val}.")

        tensors = tuple(torch.cat(t) for t in zip(*to_concat['tensors']))

        return cls(
            *tensors,
            group_names=np.concatenate(to_concat['group_names']),
            start_times=np.concatenate(to_concat['start_times']),
            measures=fixed['measures'],
            dt_unit=fixed['dt_unit']
        )

    def to_dataframe(self,
                     group_colname: str = 'group',
                     time_colname: str = 'time'
                     ) -> 'DataFrame':

        return self.tensor_to_dataframe(
            tensor=torch.cat(self.tensors, 2),
            times=self.times(),
            group_names=self.group_names,
            group_colname=group_colname,
            time_colname=time_colname,
            measures=self.all_measures
        )

    @staticmethod
    def tensor_to_dataframe(tensor: Tensor,
                            times: np.ndarray,
                            group_names: Sequence,
                            group_colname: str,
                            time_colname: str,
                            measures: Sequence[str]) -> 'DataFrame':
        from pandas import DataFrame, concat

        tensor = tensor.data.numpy()

        dfs = []
        for g, group_name in enumerate(group_names):
            # get values, don't store trailing nans:
            values = tensor[g]
            all_nan_per_row = np.min(np.isnan(values), axis=1)
            if all_nan_per_row.all():
                warn(f"Group {group_name} has only missing values.")
                continue
            end_idx = np.max(np.where(~all_nan_per_row)[0]) + 1
            # convert to dataframe:
            df = DataFrame(data=values[:end_idx, :], columns=measures)
            df[group_colname] = group_name
            df[time_colname] = times[g, 0:len(df.index)]
            dfs.append(df)

        return concat(dfs)

    @classmethod
    def from_dataframe(cls,
                       dataframe: 'DataFrame',
                       group_colname: str,
                       time_colname: str,
                       measure_colnames: Sequence[str],
                       dt_unit: Optional[str]) -> 'TimeSeriesDataset':
        assert isinstance(group_colname, str)
        assert isinstance(time_colname, str)
        assert isinstance(measure_colnames, (list, tuple))
        assert len(measure_colnames) == len(set(measure_colnames))

        # sort by time:
        dataframe = dataframe.sort_values(time_colname)

        for measure_colname in measure_colnames:
            if measure_colname not in dataframe.columns:
                raise ValueError(f"'{measure_colname}' not in dataframe.columns:\n{dataframe.columns}'")

        # first pass for info:
        arrays, time_idxs, group_names, start_times = [], [], [], []
        for g, df in dataframe.groupby(group_colname, sort=True):
            # group-names:
            group_names.append(g)

            # times:
            times = df[time_colname].values
            assert len(times) == len(set(times)), f"Group {g} has duplicate times"
            min_time = times[0]
            start_times.append(min_time)
            if dt_unit is None:
                time_idx = (times - min_time).astype('int64')
            else:
                time_idx = (times - min_time).astype(f'timedelta64[{dt_unit}]').view('int64')
            time_idxs.append(time_idx)

            # values:
            arrays.append(df.loc[:, measure_colnames].values)

        # second pass organizes into tensor
        time_len = max(time_idx[-1] + 1 for time_idx in time_idxs)
        tens = torch.empty((len(arrays), time_len, len(measure_colnames)))
        tens[:] = np.nan
        for i, (array, time_idx) in enumerate(zip(arrays, time_idxs)):
            tens[i, time_idx, :] = Tensor(array)

        return cls(
            tens,
            group_names=group_names,
            start_times=start_times,
            measures=[measure_colnames],
            dt_unit=dt_unit
        )

    def with_new_tensors(self, *tensors: Tensor) -> 'TimeSeriesDataset':
        """
        Create a new Batch with a different Tensor, but all other attributes the same.
        """
        return type(self)(
            *tensors,
            group_names=self.group_names,
            start_times=self.start_times,
            measures=self.measures,
            dt_unit=self.dt_unit
        )

    # Private ------------------------
    def _validate_start_times(self, start_times: Union[np.ndarray, Sequence], dt_unit: Optional[str]) -> np.ndarray:
        if not isinstance(start_times, np.ndarray):
            if isinstance(start_times[0], np.datetime64):
                start_times = np.array(start_times, dtype='datetime64')
            else:
                start_times_int = np.array(start_times, dtype=np.int64)
                if not np.isclose(start_times_int - start_times, 0.).all():
                    raise ValueError("`start_times` should be a datetime64 array or an array of whole numbers")
                start_times = start_times_int

        if dt_unit in self.supported_dt_units:
            start_times = start_times.astype(f"datetime64[{dt_unit}]")
        elif dt_unit == 'W':
            weekdays = set(day_of_week_num(start_times))
            if len(weekdays) > 1:
                raise ValueError(f"For weekly data, all start_times must be same day-of-week. Got:\n{weekdays}")
            # need to keep daily due how numpy does rounding: https://github.com/numpy/numpy/issues/12404
            start_times = start_times.astype('datetime64[D]')
        elif dt_unit is None:
            if not start_times.dtype == np.int64:
                raise ValueError("If `dt_unit` is None, expect start_times to be an array w/dtype of int64.")
        else:
            raise ValueError(f"Time-unit of {dt_unit} not currently supported; please report to package maintainer.")

        return start_times


# assumed by `day_of_week_num`:
assert np.zeros(1).astype('datetime64[D]') == np.datetime64('1970-01-01', 'D')


def day_of_week_num(dts: np.ndarray) -> np.ndarray:
    return (dts.astype('datetime64[D]').view('int64') - 4) % 7
