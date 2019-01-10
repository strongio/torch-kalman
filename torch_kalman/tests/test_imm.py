from torch.optim import Adam

#from torch_kalman.design.imm_design import SimpleIMMDesign
from torch_kalman.kalman_filter import KalmanFilter
#from torch_kalman.kalman_filter.imm import IMM
from torch_kalman.process import LocalTrend
from torch_kalman.tests import TestCaseTK

import numpy as np


class TestIMM(TestCaseTK):
    @staticmethod
    def make_data(seed=42):
        np.random.seed(seed=seed)
        velocity = [np.random.randn() * .10]
        for i in range(1, 100):
            velocity.append(velocity[-1] * .95 + .10 * np.random.randn())

        return np.cumsum(velocity)
    #
    # @staticmethod
    # def design_kwargs():
    #     lt = LocalTrend(id='local_trend', decay_position=False, decay_velocity=False)
    #     lt.add_measure('measured')
    #     return dict(processes=[lt], measures=['measured'])
    #
    # def test_imm_one_model(self):
    #     """
    #     TODO: test that IMM w/o any extra models gives same results as vanilla filter
    #     """
    #     pass
    #
    # def test_update(self):
    #     """
    #     TODO: test that straight causes low proc-var model to become likely
    #     TODO: test that turn causes high proc-var model to become likely
    #     """
    #     pass
    #
    # def test_imm_two_model(self):
    #     """
    #     TODO: test that a two-model IMM can get arbitrarily low error on a pre-specified dataset
    #     """
    #     # design.add_model('high_var')
    #     # design.add_process_mod(model_name='high_var',
    #     #                        process_name='local_trend',
    #     #                        state_elements=['position'],
    #     #                        init_offset=1.0)s
    #     pass
