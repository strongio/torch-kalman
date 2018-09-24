import torch
from torch import Tensor

from torch_kalman.state_belief import Gaussian
from torch_kalman.tests import TestCaseTK, simple_mv_velocity_design


class TestStateBelief(TestCaseTK):
    def test_update(self):
        design = simple_mv_velocity_design(dims=1)
        batch_design = design.for_batch(batch_size=1, time=0)

        # make data w/ large value
        data = Tensor([1000.])[:, None, None]

        # initialize belief to zeros
        sb = Gaussian(means=torch.zeros((1, 2)), covs=torch.ones((1, 2, 2)))

        # call update
        sb.compute_measurement(H=batch_design.H(), R=batch_design.R())
        update1 = sb.update(obs=data[:, 0, :])

        # try again, but override measurement-variance to be higher
        sb2 = Gaussian(means=torch.zeros((1, 2)), covs=torch.ones((1, 2, 2)))
        sb2.compute_measurement(H=batch_design.H(), R=2 * batch_design.R())
        update2 = sb2.update(obs=data[:, 0, :])

        self.assertTrue((update2.means < update1.means).all())

    def test_predict(self):
        # TODO
        pass

    def test_log_likelihood(self):
        # TODO
        pass
