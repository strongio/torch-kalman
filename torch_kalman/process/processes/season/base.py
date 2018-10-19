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

    def __init__(self,
                 id: str,
                 state_elements: Sequence[str],
                 transitions: Dict[str, Dict[str, Union[float, None]]],
                 season_start: Optional[str] = None,
                 timestep_interval: Optional[str] = None):

        # parse date information:
        if season_start is None:
            warn("`season_start` was not passed; will assume all groups start in same season.")
            self.start_datetime = None
        else:
            assert timestep_interval is not None, "If passing `season_start` must also pass `timestep_interval`."
            self.start_datetime = np.datetime64(season_start, (timestep_interval, 1))

        super().__init__(id=id, state_elements=state_elements, transitions=transitions)

        # expected for_batch kwargs:
        self.expected_batch_kwargs = ['time']
        if self.start_datetime:
            self.expected_batch_kwargs.append('start_datetimes')

    def check_datetimes(self, datetimes: np.ndarray) -> None:
        expected_dtype = self.start_datetime.dtype
        if datetimes.dtype != expected_dtype:
            raise ValueError(f"The datetimes dtype should be '{expected_dtype}'.")

    def initial_state(self,
                      batch_size: int,
                      start_datetimes: Optional[np.ndarray] = None,
                      time: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def parameters(self) -> Generator[Parameter, None, None]:
        raise NotImplementedError

    def covariance(self) -> Covariance:
        raise NotImplementedError
