import datetime
import itertools

from typing import Sequence, Any, Union, Optional, Tuple
from warnings import warn

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset

from torch_kalman.internals.repr import NiceRepr
from torch_kalman.internals.utils import ragged_cat, true1d_idx
from torch_kalman.utils.datetime import DateTimeHelper


class TimeSeriesDataset(NiceRepr, TensorDataset):
    """
    TimeSeriesDataset includes additional information about each of the Tensors' dimensions: the name for each group in
    the first dimension, the start (date)time (and optionally datetime-unit) for the second dimension, and the name of
    the measures for the third dimension.

    Note that unlike TensorDataset, indexing a TimeSeriesDataset returns another TimeSeriesDataset, not a tuple of
    tensors. So when using TimeSeriesDataset, use `TimeSeriesDataLoader` (or just use
    `DataLoader(collate_fn=TimeSeriesDataset.collate)`).
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
        self._dt_helper = DateTimeHelper(dt_unit=dt_unit, start_datetime=None)
        self.start_times = self._dt_helper.validate_datetimes(start_times)
        self.dt_unit = dt_unit
        super().__init__(*tensors)

    @property
    def sizes(self) -> Sequence:
        return [t.size() for t in self.tensors]

    # Subsetting ------------------------:
    def train_val_split(self,
                        train_frac: float = None,
                        dt: np.datetime64 = None) -> Tuple['TimeSeriesDataset', 'TimeSeriesDataset']:
        """
        :param train_frac: The proportion of the data to keep for training. This is calculated on a per-group basis, by
        taking the last observation for each group (i.e., the last observation that a non-nan value on any measure). If
        neither `train_frac` nor `dt` are passed, `train_frac=.75` is used.
        :param dt: A datetime to use in dividing train/validation (first datetime for validation).
        :return: Two TimeSeriesDatasets, one with data before the split, the other with >= the split.
        """

        # get split times:
        if dt is None:
            if train_frac is None:
                train_frac = .75
            assert 0 < train_frac < 1
            # for each group, find the last non-nan, take `frac` of that to find the train/val split point:
            split_idx = np.array([int(idx * train_frac) for idx in self._last_measured_idx()], dtype='int')
            split_times = np.take(self.times(0), split_idx)
        else:
            if train_frac is not None:
                raise TypeError("Can pass only one of `train_frac`, `dt`.")
            if not isinstance(dt, np.datetime64):
                dt = np.datetime64(dt, self.dt_unit)
            split_times = np.full(shape=len(self.group_names), fill_value=dt)

        # val:
        val_dataset = self.with_new_start_times(split_times)

        # train:
        train_tensors = []
        for i, tens in enumerate(self.tensors):
            train = tens.data.clone()
            train[np.where(self.times(i) >= split_times[:, None])] = float('nan')
            not_all_nan = (~torch.isnan(train)).sum((0, 2))
            last_good_idx = true1d_idx(not_all_nan).max()
            train = train[:, :(last_good_idx + 1), :]
            train_tensors.append(train)
        train_dataset = self.with_new_tensors(*train_tensors)

        return train_dataset, val_dataset

    def with_new_start_times(self, start_times: Union[np.ndarray, Sequence]) -> 'TimeSeriesDataset':
        """
        Subset a TimeSeriesDataset so that some/all of the groups have later start times.

        :param start_times: An array/list of new datetimes.
        :return: A new TimeSeriesDataset.
        """
        new_tensors = []
        for i, tens in enumerate(self.tensors):
            times = self.times(i)
            new_tens = []
            for g, (new_time, old_times) in enumerate(zip(start_times, times)):
                if (old_times <= new_time).all():
                    raise ValueError(f"{new_time} is later than all the times for group {self.group_names[g]}")
                elif (old_times > new_time).all():
                    raise ValueError(f"{new_time} is earlier than all the times for group {self.group_names[g]}")
                new_tens.append(tens[g, true1d_idx(old_times >= new_time), :].unsqueeze(0))
            new_tens = ragged_cat(new_tens, ragged_dim=1, cat_dim=0)
            new_tensors.append(new_tens)
        return type(self)(
            *new_tensors,
            group_names=self.group_names,
            start_times=start_times,
            measures=self.measures,
            dt_unit=self.dt_unit
        )

    def get_groups(self, groups: Sequence[Any]) -> 'TimeSeriesDataset':
        """
        Get the subset of the batch corresponding to groups. Note that the ordering in the output will match the
        original ordering (not that of `group`), and that duplicates will be dropped.
        """
        group_idx = true1d_idx(np.isin(self.group_names, groups))
        return self[group_idx]

    def split_measures(self, *measure_groups) -> 'TimeSeriesDataset':
        """
        Take a dataset with one tensor, split it into a dataset with multiple tensors.

        :param measure_groups: Each argument should be be a list of measure-names, or an indexer (i.e. list of ints or
        a slice).
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

        tensors = tuple(ragged_cat(t, ragged_dim=1) for t in zip(*to_concat['tensors']))

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
        assert tensor.shape[0] == len(group_names)
        assert tensor.shape[0] == len(times)
        assert tensor.shape[1] <= times.shape[1]
        assert tensor.shape[2] == len(measures)

        dfs = []
        for g, group_name in enumerate(group_names):
            # get values, don't store trailing nans:
            values = tensor[g]
            all_nan_per_row = np.min(np.isnan(values), axis=1)
            if all_nan_per_row.all():
                warn(f"Group {group_name} has only missing values.")
                continue
            end_idx = true1d_idx(~all_nan_per_row).max() + 1
            # convert to dataframe:
            df = DataFrame(data=values[:end_idx, :], columns=measures)
            df[group_colname] = group_name
            df[time_colname] = np.nan
            df[time_colname] = times[g, 0:len(df.index)]
            dfs.append(df)

        return concat(dfs)

    @classmethod
    def from_dataframe(cls,
                       dataframe: 'DataFrame',
                       group_colname: str,
                       time_colname: str,
                       dt_unit: Optional[str],
                       measure_colnames: Optional[Sequence[str]] = None,
                       X_colnames: Optional[Sequence[str]] = None,
                       y_colnames: Optional[Sequence[str]] = None) -> 'TimeSeriesDataset':

        if measure_colnames is None:
            if X_colnames is None or y_colnames is None:
                raise ValueError("Must pass either `measure_colnames` or `X_colnames` & `y_colnames`")
            measure_colnames = list(y_colnames) + list(X_colnames)
        else:
            if X_colnames is not None or y_colnames is not None:
                raise ValueError("If passing `measure_colnames` do not pass `X_colnames` or `y_colnames`.")

        assert isinstance(group_colname, str)
        assert isinstance(time_colname, str)
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

        dataset = cls(
            tens,
            group_names=group_names,
            start_times=start_times,
            measures=[measure_colnames],
            dt_unit=dt_unit
        )

        if X_colnames is not None:
            dataset = dataset.split_measures(y_colnames, X_colnames)
            y, X = dataset.tensors
            # don't use nan-padding on the predictor tensor:
            for i, time_idx in enumerate(time_idxs):
                X[:, time_idx.max():, :] = 0.0

        return dataset

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

    # Util/Private ------------------------
    def times(self, which: Optional[int] = None) -> np.ndarray:
        """
        A 2D array of datetimes (or integers if dt_unit is None) for this dataset.

        :param which: If this dataset has multiple tensors of different number of timesteps, which should be used for
        constructing the `times` array? Defaults to the one with the most timesteps.
        :return: A 2D numpy array of datetimes (or integers if dt_unit is None).
        """
        if which is None:
            num_timesteps = max(tensor.shape[1] for tensor in self.tensors)
        else:
            num_timesteps = self.tensors[which].shape[1]
        return self._dt_helper.make_grid(self.start_times, num_timesteps)

    def datetimes(self) -> np.ndarray:
        return self.times()

    @property
    def start_datetimes(self) -> np.ndarray:
        return self.start_times

    def last_measured_times(self) -> np.ndarray:
        """
        :return: The datetimes (or integers if dt_unit is None) for the last measurement in the first tensor, where a
        measurement is any non-nan value in at least one dimension.
        """
        times = self.times(which=0)
        last_measured_idx = self._last_measured_idx()
        return np.array([t[idx] for t, idx in zip(times, last_measured_idx)], dtype=f'datetime64[{self.dt_unit}]')

    def _last_measured_idx(self) -> np.ndarray:
        """
        :return: The indices of the last measurement in the first tensor, where a measurement is any non-nan value in at
         least on dimension.
        """
        tens, *_ = self.tensors
        any_measured_bool = ~np.isnan(tens.numpy()).all(2)
        last_measured_idx = np.array(
            [np.max(true1d_idx(any_measured_bool[g]), initial=0) for g in range(len(self.group_names))],
            dtype='int'
        )
        return last_measured_idx


class TimeSeriesDataLoader(DataLoader):
    """
    This is a convenience wrapper around `DataLoader(collate_fn=TimeSeriesDataset.collate)`. Additionally, it provides
    a `from_dataframe()` classmethod so that the data-loader can be created directly from a pandas dataframe. This can
    be more memory-efficient than the alternative route of first creating a TimeSeriesDataset from a dataframe, and then
     passing that object to a data-loader.
    """

    def __init__(self, *args, **kwargs):
        kwargs['collate_fn'] = kwargs.get('collate_fn') or TimeSeriesDataset.collate
        super().__init__(*args, **kwargs)

    @classmethod
    def from_dataframe(cls,
                       dataframe: 'DataFrame',
                       group_colname: str,
                       time_colname: str,
                       dt_unit: Optional[str],
                       measure_colnames: Optional[Sequence[str]] = None,
                       X_colnames: Optional[Sequence[str]] = None,
                       y_colnames: Optional[Sequence[str]] = None,
                       **kwargs) -> 'TimeSeriesDataLoader':
        dataset = ConcatDataset(
            datasets=[
                TimeSeriesDataset.from_dataframe(
                    dataframe=df,
                    group_colname=group_colname,
                    time_colname=time_colname,
                    measure_colnames=measure_colnames,
                    X_colnames=X_colnames,
                    y_colnames=y_colnames,
                    dt_unit=dt_unit
                )
                for g, df in dataframe.groupby(group_colname)
            ]
        )
        return cls(dataset=dataset, **kwargs)
