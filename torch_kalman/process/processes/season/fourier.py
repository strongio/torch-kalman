from typing import Generator, Tuple, Optional, Union, Dict

import torch

from torch import Tensor
from torch.nn import Parameter

from torch_kalman.covariance import Covariance

from torch_kalman.process.for_batch import ProcessForBatch
from torch_kalman.process.processes.season.base import DateAware

import numpy as np

from torch_kalman.utils import itervalues_sorted_keys, split_flat
from torch_kalman.process.utils.fourier import fourier_tensor


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

        # writing measure-matrix is slow, no need to do it repeatedly:
        self.measure_cache = {}

    def initial_state(self, **kwargs) -> Tuple[Tensor, Tensor]:
        means = self.initial_state_mean_params
        covs = Covariance.from_log_cholesky(**self.initial_state_cov_params, device=self.device)
        return means, covs

    def parameters(self) -> Generator[Parameter, None, None]:
        yield self.initial_state_mean_params
        if self.cov_cholesky_log_diag is not None:
            yield self.cov_cholesky_log_diag
        if self.cov_cholesky_log_diag is not None:
            yield self.cov_cholesky_off_diag
        for param in itervalues_sorted_keys(self.initial_state_cov_params):
            yield param

    def covariance(self) -> Covariance:
        if self.cov_cholesky_log_diag is not None:
            return Covariance.from_log_cholesky(log_diag=self.cov_cholesky_log_diag,
                                                off_diag=self.cov_cholesky_off_diag,
                                                device=self.device)
        else:
            ns = self.K * 2
            cov = torch.empty(size=(ns, ns), device=self.device)
            cov[:] = 0.
            return cov

    # noinspection PyMethodOverriding
    def add_measure(self, measure: str) -> None:
        for state_element in self.state_elements:
            super().add_measure(measure=measure, state_element=state_element, value=None)

    def for_batch(self, input: Tensor, start_datetimes: Optional[np.ndarray] = None) -> ProcessForBatch:
        # super:
        for_batch = super().for_batch(input)

        # determine the delta (integer time accounting for different groups having different start datetimes)
        delta = self.get_delta(for_batch.num_groups, for_batch.num_timesteps, start_datetimes=start_datetimes)

        # determine season:
        season = delta % self.seasonal_period

        # generate the fourier tensor:
        fourier_tens = fourier_tensor(time=Tensor(season), seasonal_period=self.seasonal_period, K=self.K)

        for measure in self.measures():
            for state_element in self.state_elements:
                r, c = (int(x) for x in state_element.split(sep=","))
                values = split_flat(fourier_tens[:, :, r, c], dim=1)
                for_batch.add_measure(measure=measure, state_element=state_element, values=values)

        return for_batch
