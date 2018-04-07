from collections import OrderedDict

from torch_kalman.design.nn_output import NNOutput
from torch_kalman.design.design_matrix import F, H, R, Q, B


# noinspection PyPep8Naming
class Design(object):
    def __init__(self, states, measurements, transition_nn=None, measurement_nn=None):
        """
        This creates the four design-matrices needed for a kalman filter:
        * R - This is a covariance matrix for the measurement noise. It's generated from the list of measurements.
        * Q - This is a covariance matrix for the process noise. It's generated from the list of states.
        * F - This is a matrix which takes the states at T_n and generates the states at T_n+1. It's generated from the
              `transitions` attribute of the states.
        * H - This is a matrix which takes the states and converts them into the observable data.

        These matrices are pytorch Variables, so if the std_dev or correlations passed to the States and Measurements
        are pytorch Parameters, you end up with design-matrices that can be optimized using pytorch's backwards method.

        :param states: A list of States
        :param measurements: A list of Measurements
        """

        # states:
        self.states = {}
        for state in states:
            if state.id in self.states.keys():
                raise ValueError("The state_id '{}' appears more than once.".format(state.id))
            else:
                self.states[state.id] = states
        # TODO: convert to ordered dict and sort keys
        [state.torchify() for state in self.states.values()]

        # measurements:
        self.measurements = {}
        for measurement in measurements:
            if measurement.id in self.measurements.keys():
                raise ValueError("The measurement_id '{}' appears more than once.".format(measurement.id))
            else:
                self.measurements[measurement.id] = measurements
        # TODO: convert to ordered dict and sort keys
        [measurement.torchify() for measurement in self.measurements.values()]

        # design-mats:
        self.F = F(states=states, nn_module=transition_nn)
        self.H = H(states=states, measurements=measurements, nn_module=measurement_nn)
        self.Q = Q(states=states)
        self.R = R(measurements=measurements)

    def reset(self):
        """
        Reset the design matrices. For efficiency, the code to generate these matrices isn't executed every time
        they are called; instead, it's executed once then the results are saved. But calling pytorch's backward method
        will clear the graph and so these matrices will need to be re-generated. So this function is called at the end
        of the forward pass computations, so that the matrices will be rebuild next time it's called.
        """
        self.F.reset()
        self.H.reset()
        self.Q.reset()
        self.R.reset()

