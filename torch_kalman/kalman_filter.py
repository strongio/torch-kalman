import torch
from torch.autograd import Variable

from torch_kalman.utils.torch_utils import expand


from IPython.core.debugger import Pdb
pdb = Pdb()

# noinspection PyPep8Naming
class KalmanFilter(torch.nn.Module):
    def __init__(self, horizon=1):
        """

        :param horizon: An integer indicating how many timesteps ahead predictions from `forward` should be for.
        """
        super(KalmanFilter, self).__init__()
        self.horizon = horizon
        self.design = None

    @property
    def num_states(self):
        return len(self.design.states)

    @property
    def num_measurements(self):
        return len(self.design.measurements)

    @property
    def state_ids(self):
        return (state.id for state in self.design.states)

    @property
    def measurement_ids(self):
        return (measurement.id for measurement in self.design.measurements)

    # Main Forward-Pass Methods --------------------
    def _filter(self, input, initial_state=None, init_nn_input=None, n_ahead=None):
        if n_ahead is None:
            n_ahead = self.horizon

        num_series, num_timesteps, num_measurements = input.data.shape

        # add an extra n_ahead dimension b/c it's needed for k_*_next, but remember this so we can remove it later
        if n_ahead == 0:
            n_ahead_was_zero = True
            n_ahead = 1
        else:
            n_ahead_was_zero = False

        # preallocate
        mean_out = [[] for _ in range(n_ahead + 1)]
        cov_out = [[] for _ in range(n_ahead + 1)]

        # initial values:
        state_mean, state_cov = self.initialize_state(initial_state, init_nn_input, num_series, num_measurements)

        # run filter:
        for t in range(-self.horizon, num_timesteps):
            # update: incorporate observed data (and state-nn predictions)
            if t >= 0:
                # if there's a nn module predicting certain components of the state, that is applied here:
                state_mean = self.design.state_nn_update(input[:, t, :], state_mean)
                # now do update step:
                state_mean, state_cov = self.kf_update(input[:, t, :], state_mean, state_cov)

            # predict
            for nh in range(n_ahead + 1):
                assign_idx = t + nh
                if assign_idx < num_timesteps:  # output only includes timepoints in original input

                    if nh > 0:
                        # if ahead is in the future, need to predict...
                        state_mean, state_cov = self.kf_predict(input[:, assign_idx, :], state_mean, state_cov)
                    # ...otherwise will just use posterior for the current time

                    if nh == 1:
                        # if ahead is 1-step ahead, save it for the next iter
                        state_mean_next, state_cov_next = state_mean, state_cov

                    if assign_idx >= 0:  # output only includes timepoints in original input
                        mean_out[nh].append(state_mean)
                        cov_out[nh].append(state_cov)

            # next timestep:
            # noinspection PyUnboundLocalVariable
            state_mean, state_cov = state_mean_next, state_cov_next

        # forward-pass is done, so make sure design-mats will be re-instantiated next time:
        self.design.reset()

        # we added an extra n_ahead dimension b/c it was needed for k_*_next. but they didn't ask for that, so remove
        # from final output:
        if n_ahead_was_zero:
            mean_out.pop()
            cov_out.pop()

        return mean_out, cov_out

    def initialize_state(self, initial_state, init_nn_input, num_series, num_measurements):
        if initial_state is None:
            if init_nn_input is None:
                if self.design.Init.nn_module:
                    raise ValueError("(Some of) the initial values in your kalman-filter are determined by the output of a "
                                     "nn module, so you need to pass `init_nn_input` to forward.")
                init_nn_input = Variable(torch.zeros((num_series, num_measurements)))
            state_mean, state_cov = self.design.initialize_state(init_nn_input)
        else:
            state_mean, state_cov = initial_state
        return state_mean, state_cov

    def forward(self, input, initial_state=None, init_nn_input=None):
        input = self.validate_input(input)

        # run kalman-filter to get predicted state
        means_per_ahead, _ = self._filter(input=input,
                                          initial_state=initial_state,
                                          init_nn_input=init_nn_input,
                                          n_ahead=self.horizon)
        means = means_per_ahead[self.horizon]

        # get predicted measurements from state
        predicted_measurements = []
        for t, mean in enumerate(means):
            H_expanded = self.H.create_for_batch(input[:, t, :])
            predicted_measurements.append(torch.bmm(H_expanded, mean))

        return torch.cat(predicted_measurements, 0)

    def validate_input(self, input):
        if len(input.data.shape) == 2 and self.num_measurements == 1:
            # if it's a univariate series, then 2D input is permitted.
            input = input[:, :, None]
        elif len(input.data.shape) != 3:
            raise ValueError("`input` should have three-dimensions: group*time*variable. ")
        return input

    # Kalman-Smoother ------------------------------
    def smooth(self, input):
        if input is not None:
            raise NotImplementedError("Kalman-smoothing is not yet implemented.")
        return None

    # Main KF Steps -------------------------
    def kf_predict(self, obs, state_mean, state_cov):
        """
        The 'predict' step of the kalman filter.

        :param obs: The observed data for this timestep. Not needed for the predict computations themselves, but will be
        passed to the `create_for_batch` methods of the design-matrices.
        :param state_mean: The state mean (for each item in the batch).
        :param state_cov: The state covariance (for each item in the batch).
        :return: The update state_mean, state_cov.
        """
        # expand design-matrices for the batch
        F_expanded = self.F.create_for_batch(obs)
        Ft_expanded = F_expanded.permute(0, 2, 1)
        Q_expanded = self.Q.create_for_batch(obs)

        state_mean = torch.bmm(F_expanded, state_mean)
        state_cov = torch.bmm(torch.bmm(F_expanded, state_cov), Ft_expanded) + Q_expanded
        return state_mean, state_cov

    def kf_update(self, obs, state_mean, state_cov):
        """
        The 'update' step of the kalman filter.

        :param obs: The observations.
        :param state_mean: Mean
        :param state_cov: Covariance
        :return: The new (mean, covariance)
        """

        # TODO: reimplement missing values
        if (obs != obs).data.any():
            raise NotImplementedError("Missing values are WIP.")

        # expand design-matrices for the batch:
        H_expanded = self.H.create_for_batch(obs)
        R_expanded = self.R.create_for_batch(obs)

        # for the actual update, we only use the autoregressive part of the input:
        kf_obs = obs[:, 0:self.design.num_measurements]

        # residual:
        residual = kf_obs - torch.bmm(H_expanded, state_mean).squeeze(2)

        # kalman-gain:
        K = self.kalman_gain(state_cov, H_expanded, R_expanded)

        # update mean and covariance:
        # x_new = state_mean.clone()
        # # from group*one_for_all_states to group*state to group*state*1:
        # is_nan_slice_x = is_nan_per_slice[:, None].expand(bs_orig, self.num_states)[:, :, None]
        # x_new[is_nan_slice_x == 0] = x_nm + torch.bmm(K, residual.unsqueeze(2))
        #
        # P_new = state_cov.clone()
        # # group*state*1 to group*state*state:
        # is_nan_slice_P = is_nan_slice_x.expand(bs_orig, self.num_states, self.num_states)
        # P_new[is_nan_slice_P == 0] = self.covariance_update(P_nm, K, H_expanded, R_expanded)
        state_mean = state_mean + torch.bmm(K, residual.unsqueeze(2))
        state_cov = self.covariance_update(state_cov, K, H_expanded, R_expanded)

        return state_mean, state_cov

    # @staticmethod
    # def nan_remove(obs, x, P):
    #     """
    #     Remove group-slices from x and P where any variables in that slice are missing.
    #
    #     :param obs: The observed data, potentially with missing values.
    #     :param x: Mean.
    #     :param P: Covariance
    #     :return: A tuple with four elements. The first three are the tensors originally passed as arguments, but without
    #      any group-slices that have missing values. The fourth is a byte-tensor indicating which slices in the original
    #      `obs` had missing-values.
    #     """
    #     bs = P.data.shape[0]  # batch-size including missings
    #     rank = P.data.shape[1]
    #     is_nan_el = (obs != obs)  # by element
    #     if is_nan_el.data.all():
    #         return None, None, None, None
    #     if is_nan_el.data.any():
    #         # get a list, one for each group, indicating 1 for nan (for *any* variable):
    #         is_nan_list = (torch.sum(is_nan_el, 1) > 0).data.tolist()
    #         # keep only the group-slices without nans:
    #         obs_nm = torch.stack([obs[i] for i, is_nan in enumerate(is_nan_list) if is_nan == 0], 0)
    #         x_nm = torch.stack([x[i] for i, is_nan in enumerate(is_nan_list) if is_nan == 0], 0)
    #         P_nm = torch.stack([P[i] for i, is_nan in enumerate(is_nan_list) if is_nan == 0], 0)
    #     else:
    #         # don't need to do anything:
    #         obs_nm, x_nm, P_nm = obs, x, P
    #
    #     # this will be used later for assigning to x/P _new.
    #     is_nan_per_slice = (torch.sum(is_nan_el, 1) > 0)
    #
    #     return obs_nm, x_nm, P_nm, is_nan_per_slice

    @staticmethod
    def kalman_gain(P, H_expanded, R_expanded):
        bs = P.data.shape[0]  # batch-size
        Ht_expanded = H_expanded.permute(0, 2, 1)
        S = torch.bmm(torch.bmm(H_expanded, P), Ht_expanded) + R_expanded  # total covariance
        Sinv = torch.cat([torch.inverse(S[i, :, :]).unsqueeze(0) for i in range(bs)], 0)  # invert, batchwise
        K = torch.bmm(torch.bmm(P, Ht_expanded), Sinv)  # kalman gain
        return K

    @staticmethod
    def covariance_update(state_cov, K, H_expanded, R_expanded):
        """
        "Joseph stabilized" covariance correction.

        :param state_cov: State covariance (for each item in the batch).
        :param K: Kalman-gain.
        :param H_expanded: The H design-matrix, expanded for each batch.
        :param R_expanded: The R design-matrix, expanded for each batch.
        :return: The new process covariance.
        """
        rank = H_expanded.data.shape[2]
        bs = state_cov.data.shape[0] # batch size
        I = Variable(expand(torch.eye(rank, rank), bs))
        p1 = (I - torch.bmm(K, H_expanded))
        p2 = torch.bmm(torch.bmm(p1, state_cov), p1.permute(0, 2, 1))
        p3 = torch.bmm(torch.bmm(K, R_expanded), K.permute(0, 2, 1))
        return p2 + p3

    # Design-Matrices ---------------------------
    @property
    def F(self):
        return self.design.F

    @property
    def H(self):
        return self.design.H

    @property
    def R(self):
        return self.design.R

    @property
    def Q(self):
        return self.design.Q
