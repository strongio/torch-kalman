from typing import Generator, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_kalman.covariance import Covariance
from torch_kalman.process import Process


class Velocity(Process):
    def __init__(self, id: str):
        # state-elements:
        state_elements = ['position', 'velocity']

        # transitions:
        transitions = {'position': {'position': 1.0, 'velocity': 1.0},
                       'velocity': {'velocity': 1.0}}

        # process-covariance:
        ns = len(state_elements)
        self.cholesky_log_diag = Parameter(data=torch.randn(ns))
        self.cholesky_off_diag = Parameter(data=torch.randn(int(ns * (ns - 1) / 2)))

        # initial state:
        self.initial_state_mean_params = Parameter(torch.randn(ns))
        self.initial_state_cov_params = dict(log_diag=Parameter(data=torch.randn(ns)),
                                             off_diag=Parameter(data=torch.randn(int(ns * (ns - 1) / 2))))

        # super:
        super().__init__(id=id, state_elements=state_elements, transitions=transitions)

    def parameters(self) -> Generator[Parameter, None, None]:
        yield self.cholesky_log_diag
        yield self.cholesky_off_diag
        yield self.initial_state_mean_params
        for param in self.initial_state_cov_params.values():
            yield param

    def initial_state(self, batch_size: int, **kwargs) -> Tuple[Tensor, Tensor]:
        means = self.initial_state_mean_params.expand(batch_size, -1)
        covs = Covariance.from_log_cholesky(**self.initial_state_cov_params).expand(batch_size, -1, -1)
        return means, covs

    def covariance(self) -> Covariance:
        return Covariance.from_log_cholesky(log_diag=self.cholesky_log_diag,
                                            off_diag=self.cholesky_off_diag)

    def add_measure(self, measure: str, state_element: str = 'position', value: Union[float, None] = 1.0) -> None:
        super().add_measure(measure=measure, state_element=state_element, value=value)
