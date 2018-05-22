from torch_kalman.design import Design
from torch_kalman.design.process.velocity import ConstantVelocity
from torch_kalman.design.measure import Measure
from torch_kalman.kalman_filter import KalmanFilter

from torch_kalman.lazy_parameter.lazy_parameters import LogLinked

import torch
from torch.nn import Parameter


class UnivariateWithVelocity(KalmanFilter):
    def __init__(self):
        """
        This is a simple kalman filter mostly for demonstration purposes. It assumes we are tracking a single variable
        with constant velocity, and estimates the latent position and velocity. It has two tuneable parameters: the
        process noise (how much latent position/velocity can change over time) and measurement noise (how much noise
        there is in taking a measurement of the latent position).
        """
        super().__init__(design=Design())

        # parameters ---
        self.initial_position = Parameter(torch.zeros(1))
        self.log_process_std_dev = Parameter(torch.zeros(2))
        process_std_dev = LogLinked(self.log_process_std_dev)

        self.log_measure_std_dev = Parameter(torch.ones(1))
        measure_std_dev = LogLinked(self.log_measure_std_dev)

        # states ---
        process = ConstantVelocity(id_prefix=None,
                                   std_devs=self.log_process_std_dev,
                                   corr=0.,
                                   initial_position=self.initial_position)
        self.design.add_process(process)

        # measures ---
        pos_measure = Measure(id=1, std_dev=measure_std_dev)
        pos_measure.add_state_element(process.observable)
        self.design.add_measure(pos_measure)

        self.design.finalize()
