from typing import Generator, Tuple, Union

import torch

from torch import Tensor
from torch.nn import Parameter

from torch_kalman.covariance import Covariance
from torch_kalman.process import Process
from torch_kalman.process.for_batch import ProcessForBatch


class Velocity(Process):

    def __init__(self, id: str, dampened: bool = True):
        # state-elements:
        state_elements = ['position', 'velocity']

        # transitions:
        self.dampened = dampened
        self.velocity_transition_sigmoid = Parameter(torch.ones(1)) if dampened else None
        transitions = self._transitions()

        # process-covariance:
        ns = len(state_elements)
        self.log_std_devs = Parameter(-7.0 * torch.ones(2))
        self.corr_tanh = Parameter(torch.zeros(1))

        # initial state:
        self.initial_state_mean_params = Parameter(torch.zeros(ns))
        self.initial_state_cov_params = dict(log_std_devs=Parameter(data=torch.zeros(ns)),
                                             corr_tanh=Parameter(torch.zeros(1)))

        # super:
        super().__init__(id=id, state_elements=state_elements, transitions=transitions)

    def _transitions(self):
        transitions = {'position': {'position': 1.0, 'velocity': 1.0},
                       'velocity': {}}
        transitions['velocity']['velocity'] = None if self.dampened else 1.0
        return transitions

    def parameters(self) -> Generator[Parameter, None, None]:
        yield self.log_std_devs
        yield self.corr_tanh
        yield self.initial_state_mean_params
        for param in self.initial_state_cov_params.values():
            yield param
        if self.dampened:
            yield self.velocity_transition_sigmoid

    def initial_state(self, batch_size: int, **kwargs) -> Tuple[Tensor, Tensor]:
        means = self.initial_state_mean_params.expand(batch_size, -1)
        covs = Covariance.from_std_and_corr(**self.initial_state_cov_params).expand(batch_size, -1, -1)
        return means, covs

    def covariance(self) -> Covariance:
        return Covariance.from_std_and_corr(log_std_devs=self.log_std_devs, corr_tanh=self.corr_tanh)

    def add_measure(self, measure: str, state_element: str = 'position', value: Union[float, None] = 1.0) -> None:
        super().add_measure(measure=measure, state_element=state_element, value=value)

    def set_to_simulation_mode(self, scale=1.0):
        super().set_to_simulation_mode()
        self.velocity_transition_sigmoid[:] = 1.0

        # initial:
        self.initial_state_mean_params[:] = 0.
        self.initial_state_cov_params['log_std_devs'][:] = torch.tensor([1.0, -5.0])
        self.initial_state_cov_params['corr_tanh'][:] = 0.

        # process:
        self.log_std_devs[:] = torch.tensor([-2.0, -9.0])
        self.corr_tanh[:] = 0.

    def for_batch(self, batch_size: int, **kwargs) -> 'ProcessForBatch':
        for_batch = super().for_batch(batch_size=batch_size, **kwargs)
        if self.dampened:
            for_batch.set_transition(from_element='velocity',
                                     to_element='velocity',
                                     values=self.velocity_transition_sigmoid.sigmoid())
        return for_batch
