from typing import Generator, Tuple, Union

import torch

from torch import Tensor
from torch.nn import Parameter

from torch_kalman.covariance import Covariance
from torch_kalman.process import Process
from torch_kalman.process.for_batch import ProcessForBatch


class LocalTrend(Process):

    def __init__(self, id: str, decay_velocity: bool = True, decay_position: bool = False):
        # state-elements:
        state_elements = ['position', 'velocity']

        # transitions:
        transitions = {'position': {'position': None, 'velocity': 1.0},
                       'velocity': {'velocity': None}}
        if not decay_position:
            transitions['position']['position'] = 1.0
        if not decay_velocity:
            transitions['velocity']['velocity'] = 1.0
        self._transition_params = None

        # process-covariance:
        ns = len(state_elements)
        self.log_std_devs = Parameter(torch.randn(2) - 3.0)
        self.corr_arctanh = Parameter(torch.randn(1))

        # initial state:
        self.initial_state_mean_params = Parameter(torch.randn(ns))
        self.initial_state_cov_params = dict(log_std_devs=Parameter(data=torch.randn(ns) + 1.0),
                                             corr_arctanh=Parameter(torch.randn(1)))

        # super:
        super().__init__(id=id, state_elements=state_elements, transitions=transitions)

    @property
    def transition_params(self):
        if self._transition_params is None:
            self._transition_params = {}
            for state_element in ('position', 'velocity'):
                if self.transitions[state_element][state_element] is None:
                    self._transition_params[state_element] = Parameter(torch.randn(1) / 5.0)
        return self._transition_params

    def parameters(self) -> Generator[Parameter, None, None]:
        yield self.log_std_devs
        yield self.corr_arctanh
        yield self.initial_state_mean_params
        for param in self.initial_state_cov_params.values():
            yield param
        for param in self.transition_params.values():
            yield param

    def initial_state(self, batch_size: int, **kwargs) -> Tuple[Tensor, Tensor]:
        means = self.initial_state_mean_params.expand(batch_size, -1)
        covs = Covariance.from_std_and_corr(**self.initial_state_cov_params).expand(batch_size, -1, -1)
        return means, covs

    def covariance(self) -> Covariance:
        return Covariance.from_std_and_corr(log_std_devs=self.log_std_devs, corr_arctanh=self.corr_arctanh)

    def add_measure(self, measure: str, state_element: str = 'position', value: Union[float, None] = 1.0) -> None:
        super().add_measure(measure=measure, state_element=state_element, value=value)

    def set_to_simulation_mode(self, scale=1.0):
        super().set_to_simulation_mode()

        # initial:
        self.initial_state_mean_params[:] = 0.
        self.initial_state_cov_params['log_std_devs'][:] = torch.tensor([1.0, -5.0])
        self.initial_state_cov_params['corr_arctanh'][:] = 0.

        # process:
        self.log_std_devs[:] = torch.tensor([-2.0, -9.0])
        self.corr_arctanh[:] = 0.

    def for_batch(self, batch_size: int, **kwargs) -> 'ProcessForBatch':
        for_batch = super().for_batch(batch_size=batch_size, **kwargs)

        for state_element, param in self.transition_params.items():
            # gradient too flat for <.5 (transitions cause state to vanish to zero), so helpful to shift starting point:
            transition_value = torch.sigmoid(3. + .75 * param)
            for_batch.set_transition(from_element=state_element, to_element=state_element, values=transition_value)

        return for_batch
