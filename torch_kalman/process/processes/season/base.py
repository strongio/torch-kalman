from typing import Sequence, Dict, Union, Optional, Tuple, Generator
from warnings import warn

import numpy as np
from torch import Tensor
from torch.nn import Parameter

from torch_kalman.covariance import Covariance
from torch_kalman.process import Process


class DateAware(Process):
    """
    Any date-aware process, serves as a base-class for seasons without being committed to a particular seasonal structure
    (e.g., discrete, fourier, etc.).
    """
    supported_dt_units = {'Y', 'D', 'h', 'm', 's'}

    def __init__(self,
                 id: str,
                 state_elements: Sequence[str],
                 transitions: Dict[str, Dict[str, Union[float, None]]],
                 season_start: Optional[str] = None,
                 dt_unit: Optional[str] = None):
        """

        :param id: See `Process`.
        :param state_elements: See `Process`.
        :param transitions: See `Process`.
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
        else:
            assert dt_unit is not None, "If passing `season_start` must also pass `dt_unit`."
            if dt_unit in self.supported_dt_units:
                self.start_datetime = np.datetime64(season_start, (dt_unit, 1))
            elif dt_unit == 'W':
                self.start_datetime = np.datetime64(season_start, ('D', 1))
            else:
                raise ValueError(f"dt_unit {dt_unit} not currently supported")

        super().__init__(id=id, state_elements=state_elements, transitions=transitions)

        # expected for_batch kwargs:
        self.expected_batch_kwargs = []
        if self.start_datetime:
            self.expected_batch_kwargs.append('start_datetimes')

    def get_delta(self, num_groups: int, num_timesteps: int, start_datetimes: np.ndarray) -> np.ndarray:
        if start_datetimes is None:
            if self.start_datetime:
                raise ValueError("`start_datetimes` argument required.")
            delta = np.broadcast_to(np.arange(0, num_timesteps), shape=(num_groups, num_timesteps))
        else:
            self.check_datetimes(start_datetimes)
            offset = (start_datetimes - self.start_datetime).view('int64')
            if self.dt_unit == 'W':
                offset = offset / 7
                bad_freq = (np.mod(offset, 1) != 0)
                if np.any(bad_freq):
                    raise ValueError(f"start_datetimes has dates with unexpected day-of-week:\n{start_datetimes[bad_freq]}")

            delta = offset.reshape((num_groups, 1)) + np.arange(0, num_timesteps)
        return delta

    def check_datetimes(self, datetimes: np.ndarray) -> None:
        act = datetimes.dtype
        exp = self.start_datetime.dtype
        assert act == exp, f"Expected datetimes with dtype {exp}, got {act}."

    def parameters(self) -> Generator[Parameter, None, None]:
        raise NotImplementedError

    @property
    def dynamic_state_elements(self) -> Sequence[str]:
        raise NotImplementedError
