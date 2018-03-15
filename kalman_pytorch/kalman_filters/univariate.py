from kalman_pytorch.design import Design
from kalman_pytorch.design.kalman_object import Measurement, State
from kalman_pytorch.kalman_filter import KalmanFilter

from kalman_pytorch.design.lazy_parameter import LogLinked

import torch
from torch.nn import Parameter


class UnivariateWithVelocity(KalmanFilter):
    def __init__(self):
        # parameters ---
        self.log_process_std_dev = Parameter(torch.zeros(1))
        process_std_dev = LogLinked(self.log_process_std_dev)

        self.log_measurement_std_dev = Parameter(torch.zeros(1))
        measurement_std_dev = LogLinked(self.log_measurement_std_dev)

        # states ---
        states = dict()

        # position and velocity:
        states['position'] = State(id='position', std_dev=process_std_dev.with_added_lambda(lambda x: pow(.5, .5) * x))
        states['velocity'] = State(id='velocity', std_dev=process_std_dev)
        states['position'].add_correlation(states['velocity'], correlation=1.)

        # next position is just positition + velocity
        states['position'].add_transition(to_state=states['position'])
        states['velocity'].add_transition(to_state=states['position'])
        # next velocity is just current velocity:
        states['velocity'].add_transition(to_state=states['velocity'])

        # measurements ---
        measurements = dict()
        measurements['position'] = Measurement(id=1, std_dev=measurement_std_dev)
        measurements['position'].add_state(states['position'])

        design = Design(states=states.values(), measurements=measurements.values())
        super(UnivariateWithVelocity, self).__init__(design=design)
