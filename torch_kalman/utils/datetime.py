from typing import Union, Sequence, Optional

import numpy as np
import datetime


class DateTimeHelper:
    supported_dt_units = {'Y', 'D', 'h', 'm', 's'}
    default_start_dt = np.datetime64('1970-01-05')  # first monday since epoch

    def __init__(self, dt_unit: Optional[str] = None):
        self.dt_unit = dt_unit

    def make_grid(self, start_datetimes: Union[np.ndarray, Sequence], num_timesteps: int) -> np.ndarray:
        assert len(start_datetimes.shape) == 1
        datetimes = self.validate_datetimes(start_datetimes)
        offset = np.arange(0, num_timesteps)
        if self.dt_unit == 'W':
            offset *= 7
        return datetimes[:, None] + offset

    def make_delta_grid(self, start_datetimes: Union[np.ndarray, Sequence], num_timesteps: int) -> np.ndarray:
        dts = self.make_grid(start_datetimes, num_timesteps)
        if self.dt_unit is None:
            out = dts.view('int64')
        else:
            out = (dts - self.default_start_dt).view('int64')
        if self.dt_unit == 'W':
            out //= 7
        return out

    def validate_datetimes(self, datetimes: Union[np.ndarray, Sequence]) -> np.ndarray:
        if not isinstance(datetimes, np.ndarray):
            datetimes = np.array(datetimes)
        if not isinstance(datetimes.flat[0], np.datetime64):
            if isinstance(datetimes.flat[0], (datetime.date, datetime.datetime)):
                datetimes = np.array(datetimes, dtype='datetime64')
            else:
                if self.dt_unit is not None:
                    raise ValueError(
                        f"dt_unit is {self.dt_unit}, but received `datetimes` that do not appear to be datetimes"
                    )
                try:
                    datetimes_int = np.array(datetimes, dtype=np.int64)
                except TypeError as e:
                    raise TypeError("Expected datetimes to be sequence of datetimes or ints.") from e
                if not np.isclose(datetimes_int - datetimes, 0.).all():
                    raise ValueError("`datetimes` should be a datetime64 array or an array of whole numbers")
                datetimes = datetimes_int

        if self.dt_unit in self.supported_dt_units:
            datetimes = datetimes.astype(f"datetime64[{self.dt_unit}]")
            # TODO: raise error if mod != 0?
        elif self.dt_unit == 'W':
            weekdays = set(day_of_week_num(datetimes))
            if len(weekdays) > 1:
                raise ValueError(f"For weekly data, all datetimes must be same day-of-week. Got:\n{weekdays}")
            # need to keep daily due how numpy does rounding
            datetimes = datetimes.astype('datetime64[D]')
        elif self.dt_unit is None:
            if not datetimes.dtype == np.int64:
                raise ValueError("If `dt_unit` is None, expect datetimes to be an array w/dtype of int64.")
        else:
            raise ValueError(f"Time-unit of {self.dt_unit} not currently supported.")

        return datetimes


# assumed by `day_of_week_num`:
assert np.zeros(1).astype('datetime64[D]') == np.datetime64('1970-01-01', 'D')


def day_of_week_num(dts: np.ndarray) -> np.ndarray:
    return (dts.astype('datetime64[D]').view('int64') - 4) % 7
