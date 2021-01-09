from typing import List
import numpy as np
from filterpy.kalman import KalmanFilter as filterpy_KalmanFilter

import torch
from torch import jit, nn, Tensor

from torch_kalman.rewrite.covariance import Covariance
from torch_kalman.rewrite.gaussian import Gaussian
from torch_kalman.rewrite.kalman_filter import KalmanFilter
from torch_kalman.rewrite.process import LocalLevel, LocalTrend


@torch.no_grad()
def test_equations():
    data = torch.Tensor([[-5., 5., 1.]])[:, :, None]

    # make torch kf:
    torch_kf = KalmanFilter(
        processes=[LocalTrend(id='lt', decay_velocity=None).set_measure('y')],
        measures=['y']
    )
    expectedF = torch.tensor([[1., 1.], [0., 1.]])
    expectedH = torch.tensor([[1., 0.]])
    F, H, Q, R = torch_kf.script_module.get_design_mats(data, {})
    assert torch.isclose(expectedF, F).all()
    assert torch.isclose(expectedH, H).all()

    # make filterpy kf:
    filter_kf = filterpy_KalmanFilter(dim_x=2, dim_z=1)
    filter_kf.x, filter_kf.P = torch_kf.script_module.get_initial_state(data)
    filter_kf.x = filter_kf.x.detach().numpy().T
    filter_kf.P = filter_kf.P.detach().numpy().squeeze(0)
    filter_kf.Q = Q.numpy().squeeze(0)
    filter_kf.R = R.numpy().squeeze(0)
    filter_kf.F = F.numpy().squeeze(0)
    filter_kf.H = H.numpy().squeeze(0)

    # compare:
    means, covs, R, H = torch_kf(data)

    #
    filter_kf.means = []
    filter_kf.covs = []
    for t in range(data.shape[1]):
        filter_kf.means.append(filter_kf.x)
        filter_kf.covs.append(filter_kf.P)
        filter_kf.update(data[:, t, :])
        filter_kf.predict()

    assert np.isclose(means.numpy().squeeze(), np.stack(filter_kf.means).squeeze()).all()
    assert np.isclose(covs.numpy().squeeze(), np.stack(filter_kf.covs).squeeze()).all()
    print('PASSED')


if __name__ == '__main__':
    cov = Covariance(rank=3, empty_idx=[1])
    # print(cov([torch.zeros(1)]))
    ll = LocalTrend('ll')
    # print(ll([torch.ones(1)]))
    test_equations()
