from typing import Generator, Tuple, Union, Callable

import torch

from torch import Tensor
from torch.nn import Parameter

from torch_kalman.covariance import Covariance
from torch_kalman.process import Process
from torch_kalman.process.for_batch import ProcessForBatch
from torch_kalman.process.utils.bounded import Bounded
from torch_kalman.utils import itervalues_sorted_keys


class LocalTrend(Process):

    def __init__(self,
                 id: str,
                 decay_velocity: Union[bool, Tuple[float, float]] = (.95, 1.00),
                 decay_position: Union[bool, Tuple[float, float]] = False):
        # state-elements:
        state_elements = ['position', 'velocity']

        # transitions:
        transitions = {'position': {'position': None, 'velocity': 1.0},
                       'velocity': {'velocity': None}}

        self.decayed_transitions = {}
        if decay_position:
            assert not isinstance(decay_position, bool), "decay_position should be floats of bounds (or False for no decay)"
            self.decayed_transitions['position'] = Bounded(*decay_position)
            transitions['position']['position'] = None
        else:
            transitions['position']['position'] = 1.0

        if decay_velocity:
            assert not isinstance(decay_velocity, bool), "decay_velocity should be floats of bounds (or False for no decay)"
            self.decayed_transitions['velocity'] = Bounded(*decay_velocity)
            transitions['velocity']['velocity'] = None
        else:
            transitions['velocity']['velocity'] = 1.0

        # process-covariance:
        ns = len(state_elements)
        self.log_std_devs = Parameter(torch.randn(ns) - 3.0)
        self.corr_arctanh = Parameter(torch.randn(1))

        # initial state:
        self.initial_state_mean_params = Parameter(torch.randn(ns))
        self.initial_state_cov_params = dict(log_std_devs=Parameter(data=torch.randn(ns) + 1.0),
                                             corr_arctanh=Parameter(torch.randn(1)))

        # super:
        super().__init__(id=id, state_elements=state_elements, transitions=transitions)

    def parameters(self) -> Generator[Parameter, None, None]:
        yield self.log_std_devs
        yield self.corr_arctanh
        yield self.initial_state_mean_params
        for param in itervalues_sorted_keys(self.initial_state_cov_params):
            yield param
        for transition in itervalues_sorted_keys(self.decayed_transitions):
            yield transition.parameter

    def initial_state(self, **kwargs) -> Tuple[Tensor, Tensor]:
        mean = self.initial_state_mean_params
        cov = Covariance.from_std_and_corr(**self.initial_state_cov_params, device=self.device)
        return mean, cov

    def covariance(self) -> Covariance:
        return Covariance.from_std_and_corr(log_std_devs=self.log_std_devs,
                                            corr_arctanh=self.corr_arctanh,
                                            device=self.device)

    def add_measure(self, measure: str, state_element: str = 'position', value: Union[float, Callable, None] = 1.0) -> None:
        super().add_measure(measure=measure, state_element=state_element, value=value)

    def for_batch(self, num_groups: int, num_timesteps: int, **kwargs) -> 'ProcessForBatch':
        for_batch = super().for_batch(num_groups, num_timesteps, **kwargs)

        for state_element, transition in self.decayed_transitions.items():
            for_batch.set_transition(from_element=state_element, to_element=state_element, values=transition.value)

        return for_batch
