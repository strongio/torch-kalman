import math
import pdb
from typing import Tuple, List

import torch
from torch import nn, Tensor
from torch.distributions.multivariate_normal import _batch_mahalanobis


class GaussianStep(nn.Module):
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
        state_dim = mean.shape[-1]
        isnan = torch.isnan(input)
        if isnan.all():
            return mean, cov
        if isnan.any():
            nandims_by_group = torch.sum(isnan, dim=-1)
            if ((nandims_by_group > 0) & (nandims_by_group < state_dim)).any():
                raise NotImplementedError  # TODO: partial nans
            no_nan_idx = (nandims_by_group == 0).nonzero().unbind(1)  # jit doesn't support as_tuple=True
            return self._update(
                input=input[no_nan_idx], mean=mean[no_nan_idx], cov=cov[no_nan_idx], H=H[no_nan_idx], R=R[no_nan_idx]
            )
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
        rank = cov.shape[1]
        I = torch.eye(rank).expand(num_groups, -1, -1)
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
        # adapted from torch.distributions.MultivariateNormal
        diff = obs - obs_mean
        scale_tril = torch.cholesky(obs_cov)
        M = _batch_mahalanobis(scale_tril, diff)
        half_log_det = scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        return -0.5 * (float(obs.shape[-1]) * math.log(2 * math.pi) + M) - half_log_det
