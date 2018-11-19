import torch
from tqdm import tqdm

from torch_kalman.design import Design
from torch_kalman.kalman_filter import KalmanFilter
from torch_kalman.process import Season, FourierSeason, LocalTrend
from torch_kalman.simulation import Simulation
import os


def design_kwargs():
    day_in_week = Season(id='day_in_week', seasonal_period=7, season_duration=24)
    week_in_year = FourierSeason(id='week_in_year', seasonal_period=365.25 * 24, K=8)
    hour_in_day = FourierSeason(id='hour_in_day', seasonal_period=24, K=8)
    level = LocalTrend(id='level')

    processes = [day_in_week, week_in_year, hour_in_day, level]
    for process in processes:
        process.add_measure('y')

    return dict(processes=processes, measures=['y'])


def simulate(*args, **kwargs):
    design = Design(**design_kwargs(), device='cpu')
    design.requires_grad_(False)
    sim = Simulation(design=design)
    return sim.simulate(*args, **kwargs, progress=True)


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
init_gpu = torch.randn((100, 100), device='cuda')
print(init_gpu[0, 0])


@profile
def profile_train_iter(device, num_groups=100, num_timesteps=24 * 31):
    kf = KalmanFilter(**design_kwargs(), device=device)
    tens = simulate(num_groups=num_groups, num_timesteps=num_timesteps)
    tens = tens.to(device)
    pred = kf(tens, progress=True)
    pred.measurement_distribution
    loss = -pred.log_prob(tens).mean()
    loss.backward()

# @profile
# def separate_create(device, size=100, iter=100, bs=500):
#     state_cov = torch.randn((bs, size, size), device=device)
#     Q = torch.randn((1, size, size), device=device).expand(bs, -1, -1)
#     out = []
#     for i in tqdm(range(iter)):
#         F = torch.eye(size, device=device)[None, :, :].expand(bs, -1, -1)
#         Ft = F.permute(0, 2, 1)
#         out.append(torch.bmm(torch.bmm(F, state_cov), Ft) + Q)
#
#
# @profile
# def initial_create(device, size=100, iter=100, bs=500):
#     state_covs = torch.empty((bs, iter, size, size), device=device)
#     state_covs[:, 0, :, :] = torch.randn((bs, size, size), device=device)
#     F = torch.randn((1, size, size), device=device).expand(bs, -1, -1)
#     Q = torch.randn((1, size, size), device=device).expand(bs, -1, -1)
#     fake_parameter = torch.ones(1, device=device)
#     for i in tqdm(range(1, iter)):
#         F[:, :, 2] = fake_parameter
#         Ft = F.permute(0, 2, 1)
#         state_covs[:, i, :, :] = torch.bmm(torch.bmm(F, state_covs[:, i - 1, :, :]), Ft) + Q
