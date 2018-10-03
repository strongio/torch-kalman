from copy import deepcopy

from typing import TypeVar

import torch
from torch import Tensor

from torch_kalman.design import Design
from torch_kalman.state_belief import Gaussian, StateBelief


class Simulation:
    def __init__(self, design: Design):

        # freeze the design:
        design = deepcopy(design)
        for param in design.parameters():
            param.requires_grad_(requires_grad=False)
        self.design = design

    @property
    def family(self) -> TypeVar('StateBelief'):
        return Gaussian

    def simulate(self, num_groups: int, num_timesteps: int, **kwargs):
        state = self.family(*self.design.get_block_diag_initial_state(batch_size=num_groups, **kwargs))

        states = []
        for t in range(num_timesteps):
            design_for_batch = self.design.for_batch(batch_size=num_groups, time=t, **kwargs)
            # move sim forward one step:
            state = state.predict(F=design_for_batch.F(), Q=design_for_batch.Q())

            # realize the state:
            state.means = state.to_distribution().sample()
            state.covs[:] = 0.

            # measure the state:
            state.compute_measurement(H=design_for_batch.H(), R=design_for_batch.R())

            states.append(state)

        # realize the measurements:
        return self.family.concatenate_over_time(state_beliefs=states).measurement_distribution.sample()
