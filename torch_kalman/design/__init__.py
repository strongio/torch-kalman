from collections import OrderedDict

from torch_kalman.design.nn_output import NNOutput
from torch_kalman.design.design_matrix import F, H, R, Q, B, InitialState

from torch_kalman.utils.torch_utils import expand


class Design(object):
    def __init__(self, states, measurements, transition_nn=None, measurement_nn=None, init_nn=None, state_nn=None):
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
        # sort:
        self.states = OrderedDict((k, self.states[k]) for k in sorted(self.states.keys()))
        # convert any floats to Variables:
        [state.torchify() for state in self.states.values()]

        # measurements:
        self.measurements = {}
        for measurement in measurements:
            if measurement.id in self.measurements.keys():
                raise ValueError("The measurement_id '{}' appears more than once.".format(measurement.id))
            else:
                self.measurements[measurement.id] = measurements
        # sort:
        self.measurements = OrderedDict((k, self.measurements[k]) for k in sorted(self.measurements.keys()))
        # convert any floats to Variables:
        [measurement.torchify() for measurement in self.measurements.values()]

        # design-mats:
        # TODO: states/measurements are ordered-dicts or lists?
        self.F = F(states=states, nn_module=transition_nn)
        self.H = H(states=states, measurements=measurements, nn_module=measurement_nn)
        self.Q = Q(states=states)
        self.R = R(measurements=measurements)

        # initial-values:
        self.Init = InitialState(states, nn_module=init_nn)

        # if there are any NNStates:
        self.state_nn = state_nn

    def nn_modules(self):
        """
        Return all of the nn-modules that should be registered in __init__ so their parameters can be tracked.
        :return: transition_nn, measurement_nn, init_nn, state_nn
        """
        return self.F.nn_module, self.H.nn_module, self.Init.nn_module, self.state_nn

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

    def initial_values(self, batch):
        """
        Get the initial state and the initial covariance. The initial state is given by the initial values passed to each
        state, and the initial covariance is just the Q design matrix.

        :param batch: The batch these intial values will be used for.
        :return: The initial state and initial covariance, each with dimensions matching the batch-size.
        """
        bs = batch.data.shape[0]

        initial_mean_expanded = self.Init.create_for_batch(batch)

        initial_cov = self.Q.template.clone()
        initial_cov_expanded = expand(initial_cov, bs)

        return initial_mean_expanded, initial_cov_expanded
