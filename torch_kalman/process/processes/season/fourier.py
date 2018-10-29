import itertools
from math import pi
from typing import Generator, Tuple, Optional, Union

import torch

from torch import Tensor
from torch.nn import Parameter

from torch_kalman.covariance import Covariance

from torch_kalman.process.for_batch import ProcessForBatch
from torch_kalman.process.processes.season.base import DateAware

import numpy as np


class FourierSeason(DateAware):
    """
    One way of implementing a seasonal process as a fourier series. A simpler implementation than Hydnman et al., pros vs.
    cons are still TBD; please consider this experimental.
    """

    def __init__(self,
                 id: str,
                 seasonal_period: Union[int, float],
                 K: int,
                 allow_process_variance: bool = False,
                 **kwargs):
        # season structure:
        self.seasonal_period = seasonal_period
        self.K = K

        # initial state:
        ns = self.K * 2
        self.initial_state_mean_params = Parameter(torch.randn(ns))
        self.initial_state_cov_params = dict(log_diag=Parameter(data=torch.randn(ns)),
                                             off_diag=Parameter(data=torch.randn(int(ns * (ns - 1) / 2))))

        # process covariance:
        self.cov_cholesky_log_diag = Parameter(data=torch.zeros(ns)) if allow_process_variance else None
        self.cov_cholesky_off_diag = Parameter(data=torch.zeros(int(ns * (ns - 1) / 2))) if allow_process_variance else None

        #
        state_elements = []
        transitions = {}
        for r in range(self.K):
            for c in range(2):
                element_name = f"{r},{c}"
                state_elements.append(element_name)
                transitions[element_name] = {element_name: 1.0}

        super().__init__(id=id, state_elements=state_elements, transitions=transitions, **kwargs)

    def initial_state(self, batch_size: int, **kwargs) -> Tuple[Tensor, Tensor]:
        means = self.initial_state_mean_params.expand(batch_size, -1)
        covs = Covariance.from_log_cholesky(**self.initial_state_cov_params).expand(batch_size, -1, -1)
        return means, covs

    def parameters(self) -> Generator[Parameter, None, None]:
        yield self.initial_state_mean_params
        if self.cov_cholesky_log_diag is not None:
            yield self.cov_cholesky_log_diag
        if self.cov_cholesky_log_diag is not None:
            yield self.cov_cholesky_off_diag
        for param in self.initial_state_cov_params.values():
            yield param

    def covariance(self) -> Covariance:
        if self.cov_cholesky_log_diag is not None:
            return Covariance.from_log_cholesky(log_diag=self.cov_cholesky_log_diag, off_diag=self.cov_cholesky_off_diag)
        else:
            ns = self.K * 2
            cov = Covariance(size=(ns, ns))
            cov[:] = 0.
            return cov

    # noinspection PyMethodOverriding
    def add_measure(self, measure: str, state_element=None) -> None:
        """
        Rare that we we would only want a subset of the fourier components to contribute to a measurement, so allow an
        iterable of state_elements.
        """
        if state_element is None:
            state_element = [f"{r},{c}" for r, c in itertools.product(range(self.K), range(2))]
        if isinstance(state_element, str):
            super().add_measure(measure=measure, state_element=state_element, value=None)
        else:
            for se in state_element:
                super().add_measure(measure=measure, state_element=se, value=None)

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
            delta = torch.empty((batch_size,))
            delta[:] = time
        else:
            self.check_datetimes(start_datetimes)
            delta = Tensor((start_datetimes - self.start_datetime).view('int64') + time)

        for measure in self.measures():
            for state_element in self.state_elements:
                r, c = (int(x) for x in state_element.split(sep=","))
                values = 2. * pi * r * delta / self.seasonal_period
                values = torch.sin(values) if c == 0 else torch.cos(values)
                for_batch.add_measure(measure=measure, state_element=state_element, values=values)

        return for_batch
