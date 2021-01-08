from typing import Tuple, List, Optional

import torch
from torch import jit, Tensor


class StateBelief(jit.ScriptModule):
    def __init__(self,
                 mean: Tensor,
                 cov: Tensor,
                 H: Tensor,
                 R: Tensor,
                 F: Tensor,
                 Q: Tensor):
        super().__init__()
        self.mean = mean
        self.cov = cov
        self.num_groups, self.rank = mean.shape
        self.H = H.expand(self.num_groups, -1, -1) if H.ndimension() == 2 else H
        self.R = R.expand(self.num_groups, -1, -1) if R.ndimension() == 2 else R
        self.F = F.expand(self.num_groups, -1, -1) if F.ndimension() == 2 else F
        self.Q = Q.expand(self.num_groups, -1, -1) if Q.ndimension() == 2 else Q

    @jit.script_method
    def predict(self) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    @jit.script_method
    def update(self, input: Tensor) -> None:
        raise NotImplementedError

    @jit.script_method
    def forward(self, input: Tensor):
        self.update(input)
        return self.predict()


class Gaussian(StateBelief):
    @jit.script_method
    def predict(self) -> Tuple[Tensor, Tensor]:
        Ft = self.F.permute(0, 2, 1)
        mean = self.F.matmul(self.mean.unsqueeze(2)).squeeze(2)
        cov = self.F.matmul(self.cov).matmul(Ft) + self.Q
        return mean, cov

    @jit.script_method
    def update(self, input: Tensor) -> None:
        Ht = self.H.permute(0, 2, 1)
        system_cov = self.H.matmul(self.cov).matmul(Ht) + self.R
        K = self.kalman_gain(system_covariance=system_cov, Ht=Ht)
        measured_mean = self.H.matmul(self.mean.unsqueeze(2)).squeeze(2)
        new_mean = self.mean + K.matmul((input - measured_mean).unsqueeze(2)).squeeze(2)
        new_cov = self.covariance_update(K=K)
        self.mean = new_mean
        self.cov = new_cov

    @jit.script_method
    def covariance_update(self, K: Tensor) -> Tensor:
        """
        "Joseph stabilized" covariance correction.
        """
        I = torch.eye(self.rank, self.rank).expand(self.num_groups, -1, -1)
        p1 = I - K.matmul(self.H)
        p2 = p1.matmul(self.cov).matmul(p1.permute(0, 2, 1))
        p3 = K.matmul(self.R).matmul(K.permute(0, 2, 1))
        return p2 + p3

    @jit.script_method
    def kalman_gain(self, system_covariance: Tensor, Ht: Tensor) -> Tensor:
        covs_measured = self.cov.matmul(Ht)
        A = system_covariance.permute(0, 2, 1)
        B = covs_measured.permute(0, 2, 1)
        Kt, _ = torch.solve(B, A)
        K = Kt.permute(0, 2, 1)
        return K


class KalmanFilter(jit.ScriptModule):
    state_belief_cls = Gaussian

    def __init__(self,
                 H: Tensor,
                 R: Tensor,
                 F: Tensor,
                 Q: Tensor):
        super().__init__()
        self.Q = Q
        self.F = F
        self.R = R
        self.H = H
        self.state_rank = Q.shape[1]
        self.measure_rank = R.shape[1]

    @jit.script_method
    def get_initial_state(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        num_groups, *_ = input.shape
        mean = torch.zeros(num_groups, self.state_rank)
        cov = torch.eye(self.state_rank).expand(num_groups, -1, -1)
        return mean, cov

    @jit.script_method
    def forward(self, input: Tensor, state: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
        if state is None:
            mean, cov = self.get_initial_state(input)
        else:
            mean, cov = state
        inputs = input.unbind(1)
        means = torch.jit.annotate(List[Tensor], [mean])
        covs = torch.jit.annotate(List[Tensor], [cov])
        for i in range(len(inputs) - 1):
            sb = self.state_belief_cls(mean, cov, H=self.H, R=self.R, F=self.F, Q=self.Q)
            mean, cov = sb(inputs[i])
            means += [mean]
            covs += [cov]
        return torch.stack(means, 1), torch.stack(covs, 1)


from filterpy.kalman import KalmanFilter as filterpy_KalmanFilter
import numpy as np


def test_equations():
    data = Tensor([[-5., 5., 1.]])[:, :, None]
    simple_design_kwargs = {
        'F': torch.tensor([[1., 1.], [0., 1.]]),
        'H': torch.tensor([[1., 0.]]),
        'R': torch.tensor([[1.0170]]),
        'Q': torch.tensor([[0.1034, 0.0009], [0.0009, 0.1020]])
    }

    # make torch kf:
    torch_kf = KalmanFilter(**simple_design_kwargs)
    means, covs = torch_kf(data)
    print("tk:")
    print(means)
    print(covs)

    # make filterpy kf:
    filter_kf = filterpy_KalmanFilter(dim_x=2, dim_z=1)
    filter_kf.x, filter_kf.P = torch_kf.get_initial_state(data)
    filter_kf.x = filter_kf.x.detach().numpy().T
    filter_kf.P = filter_kf.P.detach().numpy().squeeze(0)
    for k, v in simple_design_kwargs.items():
        setattr(filter_kf, k, v.detach().numpy())

    # compare:
    filter_kf.means = []
    filter_kf.covs = []
    for t in range(data.shape[1]):
        filter_kf.means.append(filter_kf.x)
        filter_kf.covs.append(filter_kf.P)
        filter_kf.update(data[:, t, :])
        filter_kf.predict()
    print("filterpy:")
    print(np.stack(filter_kf.means).squeeze())
    print(np.stack(filter_kf.covs).squeeze())


if __name__ == '__main__':
    test_equations()
