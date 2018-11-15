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
        state = self.family(*self.design.get_block_diag_initial_state(batch_size=num_groups, **kwargs))

        iterator = range(num_timesteps)
        if kwargs.get('progress', None):
            from tqdm import tqdm
            iterator = tqdm(iterator)

        states = []
        for t in iterator:
            design_for_batch = self.design.for_batch(batch_size=num_groups, time=t, **kwargs)
            # move sim forward one step:
            state = state.predict(F=design_for_batch.F(), Q=design_for_batch.Q())

            # realize the state:
            state.means = state.to_distribution().sample()
            # virtually zero, but avoid numerical issues for those states w/o process covariance:
            state.covs[:] = torch.eye(self.design.state_size) * 1e-10

            # measure the state:
            state.compute_measurement(H=design_for_batch.H(), R=design_for_batch.R())

            states.append(state)

        # realize the measurements:
        return self.family.concatenate_over_time(state_beliefs=states).measurement_distribution.sample()
