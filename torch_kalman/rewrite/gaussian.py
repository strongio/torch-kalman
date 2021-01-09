from typing import Tuple

import torch
from torch import jit, Tensor


class Gaussian(jit.ScriptModule):

    @jit.script_method
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
