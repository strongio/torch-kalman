from typing import Tuple

import torch
from torch import Tensor
from torch.nn import ParameterList
from torch_kalman.design import Design
from torch_kalman.state_belief import StateBelief


class KalmanFilter(torch.nn.Module):
    def __init__(self, design: Design) -> None:
        self.design = design

        self.design_parameters = ParameterList()

        # process params:
        for process in self.design.processes:
            for param in process.parameters():
                self.design_parameters.append(param)

        # measure params:
        for param in self.design.parameters():
            self.design_parameters.append(param)

        # torch.nn.Module
        super().__init__()

    # noinspection PyShadowingBuiltins
    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        """
        :param input: A group X time X dimension Tensor.
        :return: Tuple with:
            Mean: A groups X time X dimension x 1 Tensor
            Cov: A groups X time X dimension X dimension Tensor
        """

        num_groups, num_timesteps, num_dims_all = input.shape

        state_belief = self.initialize_state_belief()

        state_beliefs = []
        for t in range(num_timesteps):
            if t >= 0:
                # update step:
                state_belief = self.kf_update(state_belief, obs=input[:, t, :])

            state_belief = self.kf_predict(state_belief, obs=input[:, t, :])
            state_beliefs.append(state_belief)

        mean, cov = zip(*[state_belief.to_tensors() for state_belief in state_beliefs])
        return torch.concat(mean), torch.concat(cov)

    def kf_predict(self, state_belief: StateBelief, obs: Tensor) -> StateBelief:
        """

        :param state_belief:
        :param obs:
        :return:
        """
        batch_design = self.design.for_batch(batch_size=len(obs))
        # batch-specific changes to design would go here
        batch_design.lock()
        state_belief_new = state_belief.predict(F=batch_design.F, Q=batch_design.Q)
        return state_belief_new

    def kf_update(self, state_belief: StateBelief, obs: Tensor) -> StateBelief:
        """

        :param state_belief:
        :param obs:
        :return:
        """
        batch_design = self.design.for_batch(batch_size=len(obs))
        # batch-specific changes to design would go here
        batch_design.lock()
        state_belief_new = state_belief.update(obs=obs, H=batch_design.H, R=batch_design.R)
        return state_belief_new

    def logliklihood(self, *args, **kwargs):
        # classmethod of state-belief?
        raise NotImplementedError("TODO")

    def initialize_state_belief(self):
        # this is where the actual type of state-belief (e.g. gaussian) is chosen
        raise NotImplementedError("TODO")
