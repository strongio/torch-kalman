from typing import Optional, Union
from warnings import warn

import numpy as np


class DTTracker:
    supported_dt_units = {'Y', 'D', 'h', 'm', 's'}

    def __init__(self,
                 season_start: Union[str, None, bool] = None,
                 dt_unit: Optional[str] = None):
        """
        :param season_start: A string that can be parsed into a datetime by `numpy.datetime64`. This is when the season
        starts, which is useful to specify if season boundaries are meaningful. It is important to specify if different
        groups in your dataset start on different dates; when calling the kalman-filter you'll pass an array of
        `start_datetimes` for group in the input, and this will be used to align the seasons for each group.
        :param dt_unit: Currently supports {'Y', 'D', 'h', 'm', 's'}. 'W' is experimentally supported.
        """

        # parse date information:
        self.dt_unit = dt_unit
        if season_start is None:
            warn("`season_start` was not passed; will assume all groups start in same season.")
            self.start_datetime = None
        elif season_start is False:
            self.start_datetime = None
        else:
            if dt_unit in self.supported_dt_units:
                self.start_datetime = np.datetime64(season_start, (dt_unit, 1))
            elif dt_unit == 'W':
                self.start_datetime = np.datetime64(season_start, ('D', 1))
            else:
                raise ValueError(f"dt_unit {dt_unit} not currently supported")
            assert dt_unit is not None, "If passing `season_start` must also pass `dt_unit`."

    def get_delta(self, num_groups: int, num_timesteps: int, start_datetimes: np.ndarray) -> np.ndarray:
        if start_datetimes is None:
            if self.start_datetime:
                raise ValueError("`start_datetimes` argument required.")
            delta = np.broadcast_to(np.arange(0, num_timesteps), shape=(num_groups, num_timesteps))
        else:

            act = start_datetimes.dtype
            exp = self.start_datetime.dtype
            assert act == exp, f"Expected datetimes with dtype {exp}, got {act}."

            offset = (start_datetimes - self.start_datetime).view('int64')
            if self.dt_unit == 'W':
                offset = offset / 7
                bad_freq = (np.mod(offset, 1) != 0)
                if np.any(bad_freq):
                    raise ValueError(f"start_datetimes has dates with unexpected day-of-week:\n{start_datetimes[bad_freq]}")

            delta = offset.reshape((num_groups, 1)) + np.arange(0, num_timesteps)
        return delta