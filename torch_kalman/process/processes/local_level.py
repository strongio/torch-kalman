from typing import Generator, Union, Tuple, Callable

import torch
from torch.nn import Parameter

from torch_kalman.process import Process
from torch_kalman.process.utils.bounded import Bounded


class LocalLevel(Process):
    def __init__(self, id: str, decay: Union[bool, Tuple[float, float]] = False):
        state_elements = ['position']

        transitions = {'position': {'position': None}}
        if decay:
            assert not isinstance(decay, bool), "decay should be floats of bounds (or False for no decay)"
            self.decay = Bounded(*decay)
        else:
            self.decay = None
            transitions['position']['position'] = 1.0

        # process-covariance:
        self.log_std_dev = Parameter(torch.randn(1) - 3.0)

        # initial state
        self.initial_state_mean_param = Parameter(torch.randn(1))
        self.initial_state_log_std_dev_param = Parameter(torch.randn(1))

        super().__init__(id=id, state_elements=state_elements, transitions=transitions)

    def parameters(self) -> Generator[Parameter, None, None]:
        yield self.log_std_dev
        yield self.initial_state_mean_param
        yield self.initial_state_log_std_dev_param
        if self.decay is not None:
            yield self.decay.parameter

    def initial_state(self, **kwargs):
        mean = self.initial_state_mean_param
        cov = torch.zeros((1, 1), device=self.device)
        cov[:] = torch.pow(torch.exp(self.initial_state_log_std_dev_param), 2)
        return mean, cov

    def covariance(self):
        cov = torch.zeros((1, 1), device=self.device)
        cov[:] = torch.pow(torch.exp(self.log_std_dev), 2)
        return cov

    def add_measure(self, measure: str, state_element: str = 'position', value: Union[float, Callable, None] = 1.0) -> None:
        super().add_measure(measure=measure, state_element=state_element, value=value)

    def for_batch(self, num_groups: int, num_timesteps: int, **kwargs) -> 'ProcessForBatch':
        for_batch = super().for_batch(num_groups, num_timesteps, **kwargs)

        if self.decay is not None:
            for_batch.set_transition(from_element='position', to_element='position', values=self.decay.value)
        return for_batch
