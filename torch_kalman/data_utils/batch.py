from typing import Sequence, Any, Optional, Union, Tuple

from pandas import DataFrame
from torch import Tensor
import numpy as np
from torch_kalman.data_utils.utils import tens_to_long

from math import floor


class Batch(tuple):
    """
    The Tensor stored by Batch is a 3D Tensor representing a batch of temporal, multivariate data. The first dimension is
    the batch dimension, the second dimension is the time dimension, and the third dimension is for each measured variable.

    This class stores the Tensor in its first position (passed via the first arg), and optional additional information about
    that Tensor in subsequent positions (passed via subsequent args).
    """

    def __new__(cls, *args):
        return super(Batch, cls).__new__(cls, args)

    def __init__(self, *args, **kwargs):
        super().__init__()

    @property
    def tensor(self):
        return self[0]

    def to_dataframe(self, tensor: Optional[Tensor] = None) -> DataFrame:
        if tensor is None:
            tensor = self.tensor

        table = tens_to_long(tensor)
        return DataFrame(table)

    def with_new_tensor(self, tensor: Tensor) -> 'Batch':
        """
        Create a new Batch with a different Tensor, but all other attributes the same.
        :param tensor:
        :return:
        """
        return self.__class__(tensor, *self[1:])

    def center(self):
        """
        :return: The mean of the tensor along the 2nd dimension (i.e., a separate mean for each group X measure).
        """
        return self.tensor.mean(1)[:, None, :]

    def scale(self):
        """
        :return: The std-dev of the tensor along the 2nd dimension (i.e., a separate std-dev for each group X measure).
        """
        return self.tensor.std(1)[:, None, :]


class TimeSeriesBatch(Batch):
    """
    TimeSeriesBatch includes additional information about each of the Tensor's dimensions: the name for each group in the
    first dimension, the start datetime and datetime-unit for the second dimension, and the name of the measures for the
    third dimension.
    """

    def __new__(cls,
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

        return super(TimeSeriesBatch, cls).__new__(cls, tensor, group_names, start_datetimes, measures)

    def subset(self, ind: Union[int, Sequence, slice]) -> 'TimeSeriesBatch':
        if isinstance(ind, int):
            ind = [ind]
        return self.__class__(self.tensor[ind], self.group_names[ind], self.start_datetimes[ind], self.measures)

    @property
    def group_names(self):
        return self[1]

    @property
    def start_datetimes(self):
        return self[2]

    @property
    def measures(self):
        return self[3]

    def datetimes(self) -> np.ndarray:
        return self.start_datetimes[:, None] + np.arange(0, self.tensor.shape[1])

    def split(self, split_frac: float) -> Tuple['TimeSeriesBatch', 'TimeSeriesBatch']:
        """
        Split data along a pre-post train/validation.
        """
        time_len = self.tensor.shape[1]
        idx = floor(time_len * split_frac)
        train_batch = self.with_new_tensor(self.tensor[:, :idx, :])
        if idx < time_len:
            val_batch = self.__class__(self.tensor[:, idx:, :],
                                       self.group_names,
                                       self.start_datetimes + idx,
                                       self.measures)
        else:
            raise ValueError("`split_frac` too large")
        return train_batch, val_batch

    def to_dataframe(self,
                     tensor: Optional[Tensor] = None,
                     group_colname: str = 'group',
                     datetime_colname: str = 'datetime',
                     measure_colname: str = 'measure',
                     value_colname: str = 'value',
                     ) -> DataFrame:
        df = super().to_dataframe(tensor=tensor)

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
