from typing import Tuple, Sequence, Optional

import torch
from torch import Tensor
from torch.distributions import Distribution

# noinspection PyPep8Naming
from torch_kalman.design import Design
from torch_kalman.design.for_batch import DesignForBatch


class StateBelief:
    def __init__(self, means: Tensor, covs: Tensor):
        """
        Belief in the state of the system at a particular timepoint, for a batch of time-series.

        :param means: The means (2D tensor)
        :param covs: The covariances (3D tensor).
        """
        assert means.dim() == 2, "mean should be 2D (first dimension batch-size)"
        assert covs.dim() == 3, "cov should be 3D (first dimension batch-size)"
        if (means != means).any():
            raise ValueError("Missing values in StateBelief (can be caused by gradient-issues -> nan initial-state).")

        batch_size, state_size = means.shape
        assert covs.shape[0] == batch_size, "The batch-size (1st dimension) of cov doesn't match that of mean."
        assert covs.shape[1] == covs.shape[2], "The cov should be symmetric in the last two dimensions."
        assert covs.shape[1] == state_size, "The state-size (2nd/3rd dimension) of cov doesn't match that of mean."

        self.batch_size = batch_size
        self.means = means
        self.covs = covs
        self._H = None
        self._R = None
        self._measurement = None

    def compute_measurement(self, H: Tensor, R: Tensor) -> None:
        if self._measurement is None:
            self._H = H
            self._R = R
        else:
            raise ValueError("`compute_measurement` has already been called for this object")

    @property
    def H(self) -> Tensor:
        if self._H is None:
            raise ValueError("This StateBelief hasn't been measured; use the `compute_measurement` method.")
        return self._H

    @property
    def R(self) -> Tensor:
        if self._R is None:
            raise ValueError("This StateBelief hasn't been measured; use the `compute_measurement` method.")
        return self._R

    @property
    def measurement(self) -> Tuple[Tensor, Tensor]:
        if self._measurement is None:
            measured_means = torch.bmm(self.H, self.means[:, :, None]).squeeze(2)
            Ht = self.H.permute(0, 2, 1)
            measured_covs = torch.bmm(torch.bmm(self.H, self.covs), Ht) + self.R
            self._measurement = measured_means, measured_covs
        return self._measurement

    def predict(self, F: Tensor, Q: Tensor) -> 'StateBelief':
        raise NotImplementedError

    def update(self, obs: Tensor) -> 'StateBelief':
        raise NotImplementedError

    @classmethod
    def concatenate_over_time(cls,
                              state_beliefs: Sequence['StateBelief'],
                              design: Optional[Design] = None) -> 'StateBeliefOverTime':
        raise NotImplementedError()

    def to_distribution(self) -> Distribution:
        raise NotImplementedError

    def simulate(self,
                 design_for_batch: DesignForBatch,
                 **kwargs) -> 'StateBeliefOverTime':

        iterator = range(design_for_batch.num_timesteps)
        if kwargs.get('progress', None):
            from tqdm import tqdm
            iterator = tqdm(iterator)

        state = self
        states = []
        for t in iterator:
            if t > 0:
                # move sim forward one step:
                state = state.predict(F=design_for_batch.F[t - 1], Q=design_for_batch.Q[t - 1])

            # realize the state:
            state.means = state.to_distribution().sample()
            # the realized state has no variance (b/c it's realized), so uncertainty will only come in on the predict step
            # from process-covariance. but *actually* no variance causes numerical issues for those states w/o process
            # covariance, so we add a small amount of variance
            state.covs[:] = torch.eye(design_for_batch.state_size) * 1e-9

            # measure the state:
            state.compute_measurement(H=design_for_batch.H[t], R=design_for_batch.R[t])

            states.append(state)

        return self.__class__.concatenate_over_time(state_beliefs=states).measurement_distribution.sample()
