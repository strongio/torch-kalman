import torch
from torch import Tensor
from torch_kalman.kalman_filter.predictions import Predictions


class Simulations(Predictions):
    """
    The output of `KalmanFilter.simulate()` -- trajectories of simulated states. Call `sample()` to convert into a
    Tensor of simulated observed-values.
    """

    def __init__(self, state_means: Tensor, H: Tensor, R: Tensor, kalman_filter: 'KalmanFilter'):
        num_groups, num_times, state_dim = state_means.shape
        super(Simulations, self).__init__(
            state_means=state_means,
            state_covs=torch.zeros((num_groups, num_times, state_dim, state_dim)),
            R=R,
            H=H,
            kalman_filter=kalman_filter
        )

    def forward(self, *args, **kwargs):
        raise NotImplementedError  # TODO: forecast doesn't make sense here... what does?

    def sample(self) -> Tensor:
        with torch.no_grad():
            dist = self.kalman_filter.kf_step.distribution_cls(self.means, self.covs)
            return dist.rsample()

    @property
    def covs(self) -> Tensor:
        return self.R

    def log_prob(self, obs: Tensor) -> Tensor:
        raise RuntimeError(f"{type(self).__name__} does not have a log-prob")
