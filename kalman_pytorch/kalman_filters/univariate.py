from kalman_pytorch.design import Design
from kalman_pytorch.design.state import position_and_velocity
from kalman_pytorch.design.measurement import Measurement
from kalman_pytorch.kalman_filter import KalmanFilter

from kalman_pytorch.design.lazy_parameter import LogLinked

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
        super(UnivariateWithVelocity, self).__init__()

        # parameters ---
        self.log_process_std_dev = Parameter(torch.zeros(1))
        process_std_dev = LogLinked(self.log_process_std_dev)

        self.log_measurement_std_dev = Parameter(torch.zeros(1))
        measurement_std_dev = LogLinked(self.log_measurement_std_dev)

        # states ---
        states = position_and_velocity(process_std_dev, '')

        # measurements ---
        pos_measurement = Measurement(id=1, std_dev=measurement_std_dev)
        pos_measurement.add_state(states[0])

        self._design = Design(states=states, measurements=[pos_measurement])

    @property
    def design(self):
        return self._design
