import itertools
from typing import Sequence, Any, Union, Optional, Tuple, Iterable
from warnings import warn

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset

from torch_kalman.internals.repr import NiceRepr
from torch_kalman.internals.utils import ragged_cat


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
        self.start_times = self._validate_start_times(start_times, dt_unit)
        self.dt_unit = dt_unit
        super().__init__(*tensors)

    def times(self) -> np.ndarray:
        num_timesteps = max(tensor.shape[1] for tensor in self.tensors)
        offset = np.arange(0, num_timesteps)
        if self.dt_unit == 'W':
            offset *= 7
        return self.start_times[:, None] + offset

    def last_measured_times(self) -> Tuple[np.ndarray, ...]:
        times = self.times()
        out = []
        for last_measured_idx in self._last_measured_idx():
            out.append(
                np.array([t[idx] for t, idx in zip(times, last_measured_idx)], dtype=f'datetime64[{self.dt_unit}]')
            )
        return tuple(out)

    def datetimes(self) -> np.ndarray:
        return self.times()

    @property
    def start_datetimes(self) -> np.ndarray:
        return self.start_times

    @property
    def sizes(self) -> Sequence:
        return [t.size() for t in self.tensors]

    # Subsetting ------------------------:
    def subset_for_training(self, train_frac: float = .75, which: int = 0) -> 'TimeSeriesDataset':
        """
        When training a forecasting model, we sometimes want to remove some proportion of the most recent data, then
        see how our model performs on the held out data. This method creates a dataset

        :param train_frac: The proportion of the data to keep for training. This is calculated on a per-group basis, by
        taking the last observation for each group (i.e., the last observation that a non-nan value on any measure).
        :param which: If the TimeSeriesDataset has multiple tensors, this subsetting is only done in one of them
        (default the first) on the assumption that other tensor(s) contain predictors, which would be needed in a
        validation period.
        :return: A TimeSeriesDataset subsetted to `train_frac` of the data.
        """
        assert 0 < train_frac < 1

        cloned_tensors = [t.data.clone() for t in self.tensors]

        # for each group, find the last non-nan, take `frac` of that to find the train/val split point:
        last_measured_idx = self._last_measured_idx()[which]
        split_idx = np.array([int(idx * train_frac) for idx in last_measured_idx], dtype='int')

        # remove excess padding:
        max_time = split_idx.max()
        cloned_tensors[which] = cloned_tensors[which][:, 0:max_time, :]

        # remove data from validation period:
        all_idx = np.broadcast_to(np.arange(0, max_time), shape=(len(split_idx), max_time))
        cloned_tensors[which][np.where(all_idx > split_idx[:, None])] = float('nan')

        return self.with_new_tensors(*cloned_tensors)

    def get_groups(self, groups: Sequence[Any]) -> 'TimeSeriesDataset':
        """
        Get the subset of the batch corresponding to groups. Note that the ordering in the output will match the
        original ordering (not that of `group`), and that duplicates will be dropped.
        """
        group_idx = np.where(np.isin(self.group_names, groups))[0]
        return self[group_idx]

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
            end_idx = np.max(np.where(~all_nan_per_row)[0]) + 1
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
            raise ValueError(f"Time-unit of {dt_unit} not currently supported.")

        return start_times

    def _last_measured_idx(self) -> Tuple[np.ndarray, ...]:
        out = []
        for tens in self.tensors:
            any_measured_bool = ~np.isnan(tens.numpy()).all(2)
            last_measured_idx = np.array(
                [np.max(np.where(any_measured_bool[g])[0], initial=0) for g in range(len(self.group_names))],
                dtype='int'
            )
            out.append(last_measured_idx)
        return tuple(out)


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


# assumed by `day_of_week_num`:
assert np.zeros(1).astype('datetime64[D]') == np.datetime64('1970-01-01', 'D')


def day_of_week_num(dts: np.ndarray) -> np.ndarray:
    return (dts.astype('datetime64[D]').view('int64') - 4) % 7
