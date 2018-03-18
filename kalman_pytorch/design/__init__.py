import torch
from torch.autograd import Variable
from kalman_pytorch.utils.torch_utils import quad_form_diag


# noinspection PyPep8Naming
class Design(object):
    def __init__(self, states, measurements):
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
        state_ids = [state.id for state in states]
        if len(state_ids) > len(set(state_ids)):
            raise ValueError("State IDs are not unique.")
        self.states = states
        [state.torchify() for state in self.states]

        measurement_ids = [measurement.id for measurement in measurements]
        if len(measurement_ids) > len(set(measurement_ids)):
            raise ValueError("measurements IDs are not unique.")
        self.measurements = measurements
        [measurement.torchify() for measurement in self.measurements]

        self._H = None
        self._R = None
        self._F = None
        self._Q = None

    def reset(self):
        """
        Reset the design matrices. For efficiency, the code to generate these matrices isn't executed every time
        they are called; instead, it's executed once then the results are saved. But calling pytorch's backward method
        will clear the graph and so these matrices will need to be re-generated. So this function is called at the end
        of the forward pass computations, so that the matrices will be rebuild next time it's called.
        """
        self._H = None
        self._R = None
        self._F = None
        self._Q = None

    @staticmethod
    def covariance_mat_from_variables(variables):
        """
        Take a list of states or measurements, and generate a covariance matrix.

        :param variables: A list of states or measurements.
        :return: A covariance Matrix, as a pytorch Variable.
        """
        num_states = len(variables)
        std_devs = Variable(torch.zeros(num_states))
        corr_mat = Variable(torch.eye(num_states))
        idx_per_var = {var.id: idx for idx, var in enumerate(variables)}
        visited_corr_idx = set()
        for var in variables:
            idx = idx_per_var[var.id]
            std_devs[idx] = var.std_dev()
            for id, corr in var.correlations.iteritems():
                idx2 = idx_per_var[id]
                if (idx, idx2) not in visited_corr_idx:
                    corr_mat[idx, idx2] = corr()
                    corr_mat[idx2, idx] = corr()
                    visited_corr_idx.add((idx, idx2))

        return quad_form_diag(std_devs, corr_mat)

    @property
    def F(self):
        if self._F is None:
            num_states = len(self.states)
            F = Variable(torch.zeros((num_states, num_states)))
            idx_per_state = {state.id: idx for idx, state in enumerate(self.states)}
            for state in self.states:
                from_idx = idx_per_state[state.id]
                for transition_to_id, multiplier in state.transitions.iteritems():
                    to_idx = idx_per_state[transition_to_id]
                    F[to_idx, from_idx] = multiplier()
            self._F = F
        return self._F

    @property
    def H(self):
        if self._H is None:
            num_measurements, num_states = len(self.measurements), len(self.states)
            H = Variable(torch.zeros((num_measurements, num_states)))
            idx_per_obs = {obs.id: idx for idx, obs in enumerate(self.measurements)}
            idx_per_state = {state.id: idx for idx, state in enumerate(self.states)}
            for measurement in self.measurements:
                for state_id, multiplier in measurement.states.iteritems():
                    obs_idx = idx_per_obs[measurement.id]
                    state_idx = idx_per_state[state_id]
                    H[obs_idx, state_idx] = multiplier()
            self._H = H
        return self._H

    @property
    def R(self):
        if self._R is None:
            self._R = self.covariance_mat_from_variables(variables=self.measurements)
        return self._R

    @property
    def Q(self):
        if self._Q is None:
            self._Q = self.covariance_mat_from_variables(variables=self.states)
        return self._Q
