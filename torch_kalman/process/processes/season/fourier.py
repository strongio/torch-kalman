from typing import Generator, Tuple, Optional

import torch
from IPython.core.debugger import Pdb
from torch import Tensor
from torch.nn import Parameter

from torch_kalman.covariance import Covariance
from torch_kalman.data_utils.utils import fourier_series
from torch_kalman.process.for_batch import ProcessForBatch
from torch_kalman.process.processes.season.base import DateAware

import numpy as np


class FixedFourierSeason(DateAware):
    """
    A low-dimensional season representation. The seasonal structure is represented as a fourier series. The shape of this
    series is fixed, and the only allowed evolution of the system is a drift in the amplitude of the fixed shape.
    """

    def __init__(self, id: str, seasonal_period: int, K: int, **kwargs):
        # initial state:
        self.initial_state_mean_param = torch.ones(1)  # Parameter(torch.randn(1) / 5.0 + 1.0)
        self.initial_state_log_std_dev_param = Parameter(torch.randn(1) - 5.)

        # process covariance:
        self.log_std_dev = -10 * torch.ones(1)  # Parameter(torch.randn(1) - 2.)

        # season structure:
        self.seasonal_period = seasonal_period
        self.K = K
        self.season_structure = Parameter(torch.randn((self.K, 2)))

        super().__init__(id=id,
                         state_elements=['scale'],
                         transitions={'scale': {'scale': 1.0}},
                         **kwargs)

    def initial_state(self, batch_size: int, **kwargs) -> Tuple[Tensor, Tensor]:
        means = torch.ones((batch_size, 1)) * self.initial_state_mean_param
        covs = Covariance(size=(batch_size, 1, 1))
        covs[:, 0, 0] = torch.pow(torch.exp(self.initial_state_log_std_dev_param), 2)
        return means, covs

    def parameters(self) -> Generator[Parameter, None, None]:
        # yield self.log_std_dev
        yield self.season_structure
        # yield self.initial_state_mean_param
        yield self.initial_state_log_std_dev_param

    def covariance(self) -> Covariance:
        cov = Covariance([[0.]])
        cov[0, 0] = torch.pow(torch.exp(self.log_std_dev), 2)
        return cov

    # noinspection PyMethodOverriding
    def add_measure(self, measure: str) -> None:
        super().add_measure(measure=measure, state_element='scale', value=None)

    def for_batch(self,
                  batch_size: int,
                  time: Optional[int] = None,
                  start_datetimes: Optional[np.ndarray] = None,
                  cache: bool = True
                  ) -> ProcessForBatch:
        for_batch = super().for_batch(batch_size=batch_size, time=time, start_datetimes=start_datetimes, cache=cache)

        if start_datetimes is None:
            if self.start_datetime:
                raise ValueError("`start_datetimes` argument required.")
            delta = torch.ones((batch_size,))
            delta[:] = time
        else:
            self.check_datetimes(start_datetimes)
            delta = Tensor((start_datetimes - self.start_datetime).view('int64') + time)

        measure_values = fourier_series(time=delta,
                                        seasonal_period=len(self.state_elements),
                                        parameters=self.season_structure)

        for measure in self.measures():
            for_batch.add_measure(measure=measure, state_element='scale', values=measure_values)

        return for_batch
