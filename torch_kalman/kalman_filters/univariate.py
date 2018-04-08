from torch_kalman.design import Design
from torch_kalman.design.process import ConstantVelocity
from torch_kalman.design.measurement import Measurement
from torch_kalman.kalman_filter import KalmanFilter

from torch_kalman.design.lazy_parameter import LogLinked

import torch
from torch.nn import Parameter
from torch.autograd import Variable


class UnivariateWithVelocity(KalmanFilter):
    def __init__(self):
        """
        This is a simple kalman filter mostly for demonstration purposes. It assumes we are tracking a single variable
        with constant velocity, and estimates the latent position and velocity. It has two tuneable parameters: the
        process noise (how much latent position/velocity can change over time) and measurement noise (how much noise
        there is in taking a measurement of the latent position).
        """
        super(UnivariateWithVelocity, self).__init__()

        # parameters ---
        self.initial_position = Parameter(torch.zeros(1))
        self.log_process_std_dev = Parameter(torch.zeros(1))
        process_std_dev = LogLinked(self.log_process_std_dev)

        self.log_measurement_std_dev = Parameter(torch.ones(1))
        measurement_std_dev = LogLinked(self.log_measurement_std_dev)

        # states ---
        process = ConstantVelocity(id_prefix=None, std_dev=process_std_dev, initial_position=self.initial_position)

        # measurements ---
        pos_measurement = Measurement(id=1, std_dev=measurement_std_dev)
        pos_measurement.add_state(process.observable)

        self.design = Design(states=process.states, measurements=[pos_measurement])
