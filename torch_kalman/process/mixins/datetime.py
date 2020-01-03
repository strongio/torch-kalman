from typing import Optional, Union
from warnings import warn

import numpy as np


class DatetimeProcess:
    supported_dt_units = {'Y', 'D', 'h', 'm', 's'}

    def __init__(self, *args, **kwargs):
        """
        :param season_start: A `numpy.datetime64` (or string that can be parsed into one). This is when the season
        starts, which is useful to specify if season boundaries are meaningful. It is important to specify if different
        groups in your dataset start on different dates; when calling the kalman-filter you'll pass an array of
        `start_datetimes` for group in the input, and this will be used to align the seasons for each group.
        :param dt_unit: Currently supports {'Y', 'D', 'h', 'm', 's'}. 'W' is experimentally supported.
        """
        dt_unit = kwargs.pop('dt_unit', None)
        season_start = kwargs.pop('season_start', None)

        # parse date information:
        if season_start is None:
            if dt_unit is not None:
                season_start = '1970-01-05'  # first monday since epoch
            else:
                raise ValueError("Must pass `dt_unit` if passing `season_start`.")

        if dt_unit in self.supported_dt_units:
            self.start_datetime = np.datetime64(season_start, (dt_unit, 1))
        elif dt_unit == 'W':
            self.start_datetime = np.datetime64(season_start, ('D', 1))
        elif dt_unit is not None:
            raise ValueError(f"dt_unit {dt_unit} not currently supported")
        else:
            self.start_datetime = None

        self.dt_unit = dt_unit

        super().__init__(*args, **kwargs)

    def _get_delta(self, num_groups: int, num_timesteps: int, start_datetimes: np.ndarray) -> np.ndarray:
        if start_datetimes is None:
            if self.start_datetime:
                raise ValueError(f"Must pass `start_datetimes` to process `{self.id}`.")
            delta = np.broadcast_to(np.arange(0, num_timesteps), shape=(num_groups, num_timesteps))
        else:

            act = start_datetimes.dtype
            exp = self.start_datetime.dtype
            if act != exp:
                raise ValueError(f"Expected datetimes with dtype {exp}, got {act}.")

            offset = (start_datetimes - self.start_datetime).view('int64')
            if self.dt_unit == 'W':
                offset = offset / 7
                bad_freq = (np.mod(offset, 1) != 0)
                if np.any(bad_freq):
                    raise ValueError(f"start_datetimes has dts with unexpected weekday:\n{start_datetimes[bad_freq]}")

            delta = offset.reshape((num_groups, 1)) + np.arange(0, num_timesteps)
        return delta
