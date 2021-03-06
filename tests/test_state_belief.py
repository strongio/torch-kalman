import unittest

import torch
from torch import Tensor

from torch_kalman.state_belief import Gaussian
from tests.utils import simple_mv_velocity_design


class TestStateBelief(unittest.TestCase):
    def test_update(self):
        design = simple_mv_velocity_design(dims=1)
        batch_design = design.for_batch(1, 1)

        # make data w/ large value
        data = Tensor([1000.])[:, None, None]

        # initialize belief to zeros
        sb = Gaussian(means=torch.zeros((1, 2)), covs=torch.ones((1, 2, 2)))

        # call update
        sb.compute_measurement(H=batch_design.H(0), R=batch_design.R(0))
        update1 = sb.update(obs=data[:, 0, :])

        # try again, but override measurement-variance to be higher
        sb2 = Gaussian(means=torch.zeros((1, 2)), covs=torch.ones((1, 2, 2)))
        sb2.compute_measurement(H=batch_design.H(0), R=2 * batch_design.R(0))
        update2 = sb2.update(obs=data[:, 0, :])

        self.assertTrue((update2.means < update1.means).all())

    def test_over_time(self):
        over_time = Gaussian.concatenate_over_time([
            Gaussian(means=torch.randn((5, 2)), covs=torch.ones((5, 2, 2))),
            Gaussian(means=torch.randn((5, 2)), covs=torch.ones((5, 2, 2))),
            Gaussian(means=torch.randn((5, 2)), covs=torch.ones((5, 2, 2))),
        ], design=None)
        self.assertEqual(over_time.num_groups, 5)
        self.assertEqual(over_time.num_timesteps, 3)
