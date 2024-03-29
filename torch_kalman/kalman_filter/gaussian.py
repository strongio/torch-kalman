from typing import Tuple, List, Dict, Optional

import torch
from torch import nn, Tensor
from torch.distributions.multivariate_normal import _batch_mahalanobis
from torch_kalman.internals.utils import get_nan_groups


class GaussianStep(nn.Module):
    """
    Used internally by `KalmanFilter` to apply the kalman-filtering algorithm. Subclasses can implement additional
    logic such as outlier-rejection, censoring, etc.
    """
    distribution_cls = torch.distributions.MultivariateNormal

    def forward(self,
                input: Tensor,
                mean: Tensor,
                cov: Tensor,
                H: Tensor,
                R: Tensor,
                F: Tensor,
                Q: Tensor) -> Tuple[Tensor, Tensor]:
        mean, cov = self.update(input, mean, cov, H=H, R=R)
        return self.predict(mean, cov, F=F, Q=Q)

    def predict(self, mean: Tensor, cov: Tensor, F: Tensor, Q: Tensor) -> Tuple[Tensor, Tensor]:
        Ft = F.permute(0, 2, 1)
        mean = F.matmul(mean.unsqueeze(2)).squeeze(2)
        cov = F.matmul(cov).matmul(Ft) + Q
        return mean, cov

    def update(self, input: Tensor, mean: Tensor, cov: Tensor, H: Tensor, R: Tensor) -> Tuple[Tensor, Tensor]:
        assert len(input.shape) > 1
        if len(input.shape) != 2:
            raise NotImplementedError

        isnan = torch.isnan(input)
        if isnan.all():
            return mean, cov
        if isnan.any():
            new_mean = mean.clone()
            new_cov = cov.clone()
            for groups, val_idx in get_nan_groups(isnan):
                if val_idx is None:
                    new_mean[groups], new_cov[groups] = self._update(
                        input=input[groups], mean=mean[groups], cov=cov[groups], H=H[groups], R=R[groups]
                    )
                else:
                    # masks:
                    m1d = torch.meshgrid(groups, val_idx)
                    m2d = torch.meshgrid(groups, val_idx, val_idx)
                    new_mean[groups], new_cov[groups] = self._update(
                        input=input[m1d[0], m1d[1]],
                        mean=mean[groups],
                        cov=cov[groups],
                        H=H[m1d[0], m1d[1]],
                        R=R[m2d[0], m2d[1], m2d[2]]
                    )
            return new_mean, new_cov
        else:
            return self._update(input=input, mean=mean, cov=cov, H=H, R=R)

    def _update(self, input: Tensor, mean: Tensor, cov: Tensor, H: Tensor, R: Tensor) -> Tuple[Tensor, Tensor]:
        # this should just be part of `update()`, but recursive calls not currently supported in torchscript
        K = self.kalman_gain(cov=cov, H=H, R=R)
        measured_mean = H.matmul(mean.unsqueeze(2)).squeeze(2)
        new_mean = mean + K.matmul((input - measured_mean).unsqueeze(2)).squeeze(2)
        new_cov = self.covariance_update(cov=cov, K=K, H=H, R=R)
        return new_mean, new_cov

    def covariance_update(self, cov: Tensor, K: Tensor, H: Tensor, R: Tensor) -> Tensor:
        """
        "Joseph stabilized" covariance correction.
        """
        num_groups = cov.shape[0]
        I = torch.eye(cov.shape[1], dtype=cov.dtype, device=cov.device).expand(num_groups, -1, -1)
        p1 = I - K.matmul(H)
        p2 = p1.matmul(cov).matmul(p1.permute(0, 2, 1))
        p3 = K.matmul(R).matmul(K.permute(0, 2, 1))
        return p2 + p3

    def kalman_gain(self, cov: Tensor, H: Tensor, R: Tensor) -> Tensor:
        Ht = H.permute(0, 2, 1)
        system_covariance = H.matmul(cov).matmul(Ht) + R
        covs_measured = cov.matmul(Ht)
        A = system_covariance.permute(0, 2, 1)
        B = covs_measured.permute(0, 2, 1)
        Kt, _ = torch.solve(B, A)
        K = Kt.permute(0, 2, 1)
        return K

    @classmethod
    def log_prob(cls, obs: Tensor, obs_mean: Tensor, obs_cov: Tensor) -> Tensor:
        return cls.distribution_cls(obs_mean, obs_cov, validate_args=False).log_prob(obs)
