from copy import deepcopy

from typing import TypeVar

from torch_kalman.design import Design
from torch_kalman.kalman_filter import KalmanFilter
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

    def initial_state(self, batch_size, **kwargs) -> StateBelief:
        means, covs = self.design.get_block_diag_initial_state(design=self.design, batch_size=batch_size, **kwargs)
        return self.family(means=means, covs=covs)

    def simulate(self, num_groups: int, num_timesteps: int, **kwargs):
        state_prediction = self.initial_state(num_groups, **kwargs)
        state_predictions = []
        for t in range(num_timesteps):
            design_for_batch = self.design.for_batch(batch_size=num_groups, time=t, **kwargs)

            state_prediction = state_prediction.predict(design_for_batch.F(), design_for_batch.Q())
            state_prediction.compute_measurement(design_for_batch.H(), design_for_batch.R())
            state_predictions.append(state_prediction)

        over_time = self.family.concatenate_over_time(state_beliefs=state_predictions)
        return over_time.measurement_distribution.sample()
