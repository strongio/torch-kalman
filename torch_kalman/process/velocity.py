from typing import Generator

import torch
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
        n = len(state_elements)
        self.cholesky_log_diag = Parameter(data=torch.randn(n))
        self.cholesky_off_diag = Parameter(data=torch.randn(int(n * (n - 1) / 2)))

        # super:
        super().__init__(id=id, state_elements=state_elements, transitions=transitions)

    def parameters(self) -> Generator[Parameter, None, None]:
        yield self.cholesky_log_diag
        yield self.cholesky_off_diag

    def covariance(self) -> Covariance:
        return Covariance.from_log_cholesky(log_diag=self.cholesky_log_diag,
                                            off_diag=self.cholesky_off_diag)

    @property
    def measurable_state(self) -> str:
        return 'position'
