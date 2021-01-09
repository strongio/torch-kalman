from typing import List
import numpy as np
from filterpy.kalman import KalmanFilter as filterpy_KalmanFilter

import torch
from torch import jit, nn, Tensor

from torch_kalman.rewrite.gaussian import Gaussian
from torch_kalman.rewrite.kalman_filter import KalmanFilter
from torch_kalman.rewrite.process import LocalLevel, LocalTrend


def test_equations():
    data = torch.Tensor([[-5., 5., 1.]])[:, :, None]
    simple_design_kwargs = {
        'F': torch.tensor([[1., 1.], [0., 1.]]),
        'H': torch.tensor([[1., 0.]]),
        'R': torch.tensor([[1.0170]]),
        'Q': torch.tensor([[0.1034, 0.0009], [0.0009, 0.1020]])
    }

    # make torch kf:
    torch_kf = KalmanFilter(
        processes=[LocalTrend(id='lt', decay_velocity=None).set_measure('y')],
        measures=['y']
    )
    torch_kf.script_module._Q = simple_design_kwargs['Q']
    torch_kf.script_module._R = simple_design_kwargs['R']
    means, covs = torch_kf(data)
    print("tk:")
    print(means)
    print(covs)

    design_check = {}
    design_check['F'], design_check['H'], design_check['Q'], design_check['R'] = \
        torch_kf.script_module.get_design_mats(data, {})

    # make filterpy kf:
    filter_kf = filterpy_KalmanFilter(dim_x=2, dim_z=1)
    filter_kf.x, filter_kf.P = torch_kf.script_module.get_initial_state(data)
    filter_kf.x = filter_kf.x.detach().numpy().T
    filter_kf.P = filter_kf.P.detach().numpy().squeeze(0)
    for k, v in simple_design_kwargs.items():
        #import pdb; pdb.set_trace()
        print(k, design_check[k])
        assert np.isclose(v, design_check[k].numpy()).all(), k
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
    #


if __name__ == '__main__':
    ll = LocalTrend('ll')
    print(ll([torch.ones(1)]))
    test_equations()
