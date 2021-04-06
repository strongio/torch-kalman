from typing import Tuple, Type, Final

import torch
from torch import nn, Tensor
from torchcast.internals.utils import get_nan_groups


class GaussianStep(nn.Module):
    """
    Used internally by `KalmanFilter` to apply the kalman-filtering algorithm. Subclasses can implement additional
    logic such as outlier-rejection, censoring, etc.
    """
    use_stable_cov_update: Final[bool] = True

    # this would ideally be a class-attribute but torch.jit.trace strips them
    @torch.jit.ignore()
    def get_distribution(self) -> Type[torch.distributions.Distribution]:
        return torch.distributions.MultivariateNormal

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
        I = torch.eye(cov.shape[1], dtype=cov.dtype, device=cov.device)[None, :, :]
        ikh = I - K @ H
        if self.use_stable_cov_update:
            return ikh @ cov @ ikh.permute(0, 2, 1) + K @ R @ K.permute(0, 2, 1)
        else:
            return ikh @ cov

    def kalman_gain(self, cov: Tensor, H: Tensor, R: Tensor) -> Tensor:
        Ht = H.permute(0, 2, 1)
        covs_measured = cov.matmul(Ht)
        system_covariance = torch.baddbmm(R, H @ cov, Ht)
        A = system_covariance.permute(0, 2, 1)
        B = covs_measured.permute(0, 2, 1)
        Kt, _ = torch.solve(B, A)
        K = Kt.permute(0, 2, 1)
        return K
