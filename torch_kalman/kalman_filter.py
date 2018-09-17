from typing import Tuple

import torch
from torch import Tensor
from torch.nn import ParameterList
from torch_kalman.design import Design
from torch_kalman.state_belief import StateBelief, Gaussian


class KalmanFilter(torch.nn.Module):
    def __init__(self, design: Design) -> None:
        super().__init__()
        self.design = design

        self.design_parameters = ParameterList()
        for param in self.design.parameters():
            self.design_parameters.append(param)

        self._state_belief = None

    @property
    def state_belief(self):
        if self._state_belief is None:
            self._state_belief = Gaussian
        return self._state_belief

    def initialize_state_belief(self, input) -> StateBelief:
        state_size = self.design.state_size()
        return self.state_belief(mean=torch.zeros(state_size).expand(len(input), -1),
                                 cov=torch.eye(state_size).expand(len(input), -1, -1))

    # noinspection PyShadowingBuiltins
    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        """
        :param input: A group X time X dimension Tensor.
        :return: Tuple with:
            Mean: A groups X time X dimension x 1 Tensor
            Cov: A groups X time X dimension X dimension Tensor
        """

        num_groups, num_timesteps, num_dims_all = input.shape

        state_belief = self.initialize_state_belief(input)

        # TODO: these are predictions *from* time t, but we want predictions *of* time t
        state_beliefs = []
        for t in range(num_timesteps):
            state_belief = self.kf_update(state_belief, obs=input[:, t, :])
            state_belief = self.kf_predict(state_belief, obs=input[:, t, :])
            state_beliefs.append(state_belief)

        mean, cov = zip(*[state_belief.to_tensors() for state_belief in state_beliefs])
        return torch.stack(mean), torch.stack(cov)

    def kf_predict(self, state_belief: StateBelief, obs: Tensor) -> StateBelief:
        batch_design = self.design.for_batch(batch_size=len(obs))
        # batch-specific changes to design would go here
        state_belief_new = state_belief.predict(F=batch_design.F(), Q=batch_design.Q())
        return state_belief_new

    def kf_update(self, state_belief: StateBelief, obs: Tensor) -> StateBelief:
        batch_design = self.design.for_batch(batch_size=len(obs))
        # batch-specific changes to design would go here
        state_belief_new = state_belief.update(obs=obs, H=batch_design.H(), R=batch_design.R())
        return state_belief_new

    def log_likelihood(self, *args, **kwargs):
        return self.state_belief.log_likelihood(*args, **kwargs)
