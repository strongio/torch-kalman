import numpy as np
from filterpy.kalman import KalmanFilter as filterpy_KalmanFilter

import torch

from torch_kalman.kalman_filter import KalmanFilter
from torch_kalman.process import LocalTrend


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
    _kwargs = torch_kf._parse_design_kwargs(input=data)
    init_state_kwargs = _kwargs.pop('init_state_kwargs')
    design_kwargs = torch_kf.script_module._get_design_kwargs_for_time(time=0, **_kwargs)
    F, H, Q, R = torch_kf.script_module.get_design_mats(num_groups=1, design_kwargs=design_kwargs, tv_kwargs=[])
    assert torch.isclose(expectedF, F).all()
    assert torch.isclose(expectedH, H).all()

    # make filterpy kf:
    filter_kf = filterpy_KalmanFilter(dim_x=2, dim_z=1)
    filter_kf.x, filter_kf.P = torch_kf.script_module.get_initial_state(data, init_state_kwargs, {})
    filter_kf.x = filter_kf.x.detach().numpy().T
    filter_kf.P = filter_kf.P.detach().numpy().squeeze(0)
    filter_kf.Q = Q.numpy().squeeze(0)
    filter_kf.R = R.numpy().squeeze(0)
    filter_kf.F = F.numpy().squeeze(0)
    filter_kf.H = H.numpy().squeeze(0)

    # compare:
    sb = torch_kf(data)

    #
    filter_kf.means = []
    filter_kf.covs = []
    for t in range(data.shape[1]):
        filter_kf.means.append(filter_kf.x)
        filter_kf.covs.append(filter_kf.P)
        filter_kf.update(data[:, t, :])
        filter_kf.predict()

    assert np.isclose(sb.means.numpy().squeeze(), np.stack(filter_kf.means).squeeze()).all()
    assert np.isclose(sb.covs.numpy().squeeze(), np.stack(filter_kf.covs).squeeze()).all()
    print('PASSED')


if __name__ == '__main__':
    test_equations()
