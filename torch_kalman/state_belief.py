from typing import Tuple

import torch
from torch import Tensor

from torch_kalman.torch_utils import batch_inverse


# noinspection PyPep8Naming
class StateBelief:
    def __init__(self, mean: Tensor, cov: Tensor):
        assert mean.dim() == 2, "mean should be 2D (first dimension batch-size)"
        assert cov.dim() == 3, "cov should be 3D (first dimension batch-size)"

        batch_size, state_size = mean.shape
        assert cov.shape[0] == batch_size, "The batch-size (1st dimension) of cov doesn't match that of mean."
        assert cov.shape[1] == cov.shape[2], "The cov should be symmetric in the last two dimensions."
        assert cov.shape[1] == state_size, "The state-size (2nd/3rd dimension) of cov doesn't match that of mean."

        self.mean = mean
        self.cov = cov

    def to_tensors(self) -> Tuple[Tensor, Tensor]:
        return self.mean, self.cov

    def predict(self, F: Tensor, Q: Tensor):
        raise NotImplementedError

    def update(self, obs: Tensor, H: Tensor, R: Tensor):
        raise NotImplementedError

    @classmethod
    def log_likelihood(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(\n\tmean={self.mean},\n\tcov={self.cov})"


# noinspection PyPep8Naming
class Gaussian(StateBelief):
    def predict(self, F: Tensor, Q: Tensor) -> StateBelief:
        Ft = F.permute(0, 2, 1)
        mean = torch.bmm(F, self.mean[:, :, None]).squeeze(2)
        cov = torch.bmm(torch.bmm(F, self.cov), Ft) + Q
        return self.__class__(mean=mean, cov=cov)

    def update(self, obs: Tensor, H: Tensor, R: Tensor) -> StateBelief:
        assert isinstance(obs, Tensor)
        # residual, kalman-gain
        residual = obs - torch.bmm(H, self.mean[:, :, None]).squeeze(2)
        K = self.kalman_gain(self.cov, H, R)

        # clone tensors since autograd can't handle in-place changes
        mean_new = self.mean.clone()
        cov_new = self.cov.clone()

        # handle kalman-update for groups w/missing values:
        isnan = (residual != residual)
        groups_with_nan = [i for i in range(len(obs)) if isnan[i].data.any()]
        if groups_with_nan:
            raise NotImplementedError("TODO: Handle missing valuees")

        # faster kalman-update for groups w/o missing values
        no_nan = [i for i in range(len(obs)) if i not in groups_with_nan]
        if len(no_nan) > 0:
            mean_new[no_nan] = self.mean[no_nan] + torch.bmm(K[no_nan], residual[no_nan].unsqueeze(2)).squeeze(2)
            cov_new[no_nan] = self.covariance_update(self.cov[no_nan], K[no_nan], H[no_nan], R[no_nan])

        return self.__class__(mean=mean_new, cov=cov_new)

    @staticmethod
    def kalman_gain(covariance, H, R):
        Ht = H.permute(0, 2, 1)
        S = torch.bmm(torch.bmm(H, covariance), Ht) + R  # total covariance
        Sinv = batch_inverse(S)
        K = torch.bmm(torch.bmm(covariance, Ht), Sinv)  # kalman gain
        return K

    @staticmethod
    def covariance_update(covariance, K, H, R):
        """
        "Joseph stabilized" covariance correction.
        """
        rank = covariance.shape[1]
        I = torch.eye(rank, rank).expand(len(covariance), -1, -1)
        p1 = (I - torch.bmm(K, H))
        p2 = torch.bmm(torch.bmm(p1, covariance), p1.permute(0, 2, 1))
        p3 = torch.bmm(torch.bmm(K, R), K.permute(0, 2, 1))
        return p2 + p3

    @classmethod
    def log_likelihood(self, *args, **kwargs):
        raise NotImplementedError("TODO")
