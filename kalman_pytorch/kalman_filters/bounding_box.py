from kalman_pytorch.design import Design
from kalman_pytorch.design.state import State
from kalman_pytorch.design.process import NoVelocity, ConstantVelocity
from kalman_pytorch.design.measurement import Measurement
from kalman_pytorch.kalman_filter import KalmanFilter

from kalman_pytorch.design.lazy_parameter import LogLinked

import torch
from torch.nn import Parameter

from collections import OrderedDict


class BoundingBox(KalmanFilter):
    def __init__(self, velocity=True, ordering=('x_left', 'y_top', 'x_right', 'y_bottom')):
        """
        This is a simple kalman filter for bounding-boxes. Note this is mostly for demonstration purposes, and is
        slightly ill-specified to 3-dimensional problems, because it tracks latent states width and height, when the
        "real" latent state for such problems should be distance-- which is an issue because the relationship between
        distance and observed width and height is non-linear (1/x) and therefore not appropriate for vanilla KFs.
        However, in practice you may find this to perform well enough.

        :param velocity: Should velocity be included in the x,y states? Defaults to True.
        :param ordering: The ordering of the input. Default is left, top, right, bottom.
        """
        super(BoundingBox, self).__init__()
        ordering_names = {'x_left', 'y_top', 'x_right', 'y_bottom'}
        if set(ordering) != ordering_names:
            raise ValueError("`ordering` must have the elements: %s" % str(ordering_names))

        self.num_states = (2 if velocity else 1) * 2 + 2

        # parameters ---
        self.log_x_process_std_dev = Parameter(torch.zeros(1))
        x_process_std_dev = LogLinked(self.log_x_process_std_dev)

        self.log_y_process_std_dev = Parameter(torch.zeros(1))
        y_process_std_dev = LogLinked(self.log_y_process_std_dev)

        self.log_height_process_std_dev = Parameter(torch.zeros(1))
        height_process_std_dev = LogLinked(self.log_height_process_std_dev)

        self.log_width_process_std_dev = Parameter(torch.zeros(1))
        width_process_std_dev = LogLinked(self.log_width_process_std_dev)

        self.log_x_measurement_std_dev = Parameter(torch.zeros(1))
        x_measurement_std_dev = LogLinked(self.log_x_measurement_std_dev)

        self.log_y_measurement_std_dev = Parameter(torch.zeros(1))
        y_measurement_std_dev = LogLinked(self.log_y_measurement_std_dev)

        self.initial_state = Parameter(torch.randn(self.num_states))
        self.initial_log_std = Parameter(torch.randn(self.num_states))

        # states ---
        process = ConstantVelocity if velocity else NoVelocity
        states = []
        states.extend(process(id_prefix='x', std_dev=x_process_std_dev).states)
        states.extend(process(id_prefix='y', std_dev=y_process_std_dev).states)
        width = State(id='width', std_dev=width_process_std_dev)
        width.add_transition(width)
        height = State(id='height', std_dev=height_process_std_dev)
        height.add_transition(height)
        states.append(width)
        states.append(height)
        states = OrderedDict((state.id, state) for state in states)  # dict for convenience

        # measurements ---
        measurements = OrderedDict()  # dict for convenience
        for measure_name in ordering:
            this_std_dev = x_measurement_std_dev if measure_name.startswith('x') else y_measurement_std_dev
            measurements[measure_name] = Measurement(id=measure_name, std_dev=this_std_dev)

        # both xs are from latent x, both ys are from latent y
        measurements['x_left'].add_state(states['x_position'])
        measurements['x_right'].add_state(states['x_position'])
        measurements['y_top'].add_state(states['y_position'])
        measurements['y_bottom'].add_state(states['y_position'])

        # as -1.*distance goes up, measured left/right and top/bottom get further apart
        measurements['x_left'].add_state(states['width'], multiplier=-.5)
        measurements['x_right'].add_state(states['width'], multiplier=.5)
        measurements['y_top'].add_state(states['height'], multiplier=-.5)
        measurements['y_bottom'].add_state(states['height'], multiplier=.5)

        self._design = Design(states=states.values(), measurements=measurements.values())

    @property
    def design(self):
        return self._design

    def initializer(self, tens):
        return self.default_initializer(tens=tens,
                                        initial_state=self.initial_state,
                                        initial_std_dev=torch.exp(self.initial_log_std))
