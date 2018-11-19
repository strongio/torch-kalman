from typing import TypeVar

import torch

from torch_kalman.design import Design
from torch_kalman.state_belief import Gaussian, StateBelief


class Simulation:
    def __init__(self, design: Design):
        assert not design.requires_grad
        self.design = design

    @property
    def family(self) -> TypeVar('StateBelief'):
        return Gaussian

    def simulate(self, num_groups: int, num_timesteps: int, **kwargs):
        design_for_batch = self.design.for_batch(input=torch.empty(num_groups, num_timesteps), **kwargs)

        iterator = range(num_timesteps)
        if kwargs.get('progress', None):
            from tqdm import tqdm
            iterator = tqdm(iterator)

        state = self.family(*design_for_batch.get_block_diag_initial_state(num_groups=num_groups))
        states = []
        for t in iterator:
            if t > 0:
                # move sim forward one step:
                F = design_for_batch.F[:, t - 1, :, :]
                Q = design_for_batch.Q[:, t - 1, :, :]
                state = state.predict(F=F, Q=Q)

            # realize the state:
            state.means = state.to_distribution().sample()
            # virtually zero, but avoid numerical issues for those states w/o process covariance:
            state.covs[:] = torch.eye(self.design.state_size) * 1e-9

            # measure the state:
            H = design_for_batch.H[:, t, :, :]
            R = design_for_batch.R[:, t, :, :]
            state.compute_measurement(H=H, R=R)

            states.append(state)

        # realize the measurements:
        return self.family.concatenate_over_time(state_beliefs=states).measurement_distribution.sample()
