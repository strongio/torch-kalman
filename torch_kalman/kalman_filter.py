import torch
from torch.autograd import Variable

from torch.nn import Parameter, ParameterList

from torch_kalman.utils.torch_utils import expand, batch_transpose, quad_form_diag

from warnings import warn


# noinspection PyPep8Naming
class KalmanFilter(torch.nn.Module):
    def __init__(self, forward_ahead=1, initializer=None, measurement_nn=None):
        """

        :param forward_ahead: An integer indicating how many timesteps ahead predictions from `forward` should be for.
        :param initializer: Optional. A callable (usually inheriting from torch.nn.Module, but can be any function) that
        takes an 2D input (supplied in `forward` as 'initializer_input') and returns (initial_mean, initial_covariance) for
        the kalman-filter. By default uses `KalmanFilter().default_initializer`.
        :param measurement_nn: Optional. A callable (virtually always inheriting from torch.nn.Module) that takes the `input`
        to `forward` and returns a 3D Variable (group * timestep * measurement). This will be added to the output of the
        kalman-filtering to generate predictions in `forward`.
        """
        super(KalmanFilter, self).__init__()
        self.forward_ahead = forward_ahead

        # measurement neural network:
        self.measurement_nn = measurement_nn

        # initial-state neural network:
        self.use_default_initializer = initializer is None
        if self.use_default_initializer:
            # need to wait to construct using `default_initializer_params`, because depends on self.num_measurements and
            # self.num_states, which aren't available until the child initializes the "design"
            self.initializer_params = None
            self.initializer = self.default_initializer

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

    def default_initializer_params(self):
        return ParameterList([Parameter(torch.zeros(self.num_measurements)), Parameter(torch.randn(self.num_states))])

    @property
    def initial_state(self):
        if not self.use_default_initializer:
            raise NotImplementedError("Overwrite the `initial_state` property if the default initializer is not used.")
        if self.initializer_params is None:
            raise ValueError("If this KalmanFilter permits the default initializer, must include the following in __init__:"
                             "`self.initializer_params = self.default_initializer_params()`.")
        return self.initializer_params[0]

    @property
    def initial_std_dev(self):
        if not self.use_default_initializer:
            raise NotImplementedError("Overwrite the `initial_std_dev` property if the default initializer is not used.")
        if self.initializer_params is None:
            raise ValueError("If this KalmanFilter permits the default initializer, must include the following in __init__:"
                             "`self.initializer_params = self.default_initializer_params()`.")
        return torch.exp(self.initializer_params[1])

    def default_initializer(self, input):
        """
        A default initializer for the kalman-filter, used when no initializer is passed. For this initializer:

        * States share their intial means with other states that contribute to the same measurement. If a state doesn't
        contribute to any measurements, or contributes to more than one, it is initialized to zero.
        * Each state gets its own initial std-deviation. All correlations are initialized to zero.

        :param input: A 2D variable whose first dimension is batch-size. For most initializers this would be used
        as an input to the nn, but for this default initializer it's simply used to get the batch-size.
        :return: Initial state-means, Initial state-covariances. Dimensions of each match the batch-size (from input).
        """
        bs = input.data.shape[0]  # batch-size

        out_mean = Variable(torch.zeros(self.num_states, 1))
        for state_idx in range(self.num_states):
            # for each state, check the measurements it contributes to:
            measurements_in_this_state = self.H[:, state_idx].data.tolist()
            if sum(measurements_in_this_state) == 1:
                # if the state contributes to one and only one measurement, it gets an init-param
                measurement_idx = measurements_in_this_state.index(1)
                out_mean[state_idx, 0] = self.initial_state[measurement_idx]
            # if state contributes to multiple or no measurements, it'll be init to zero
        # (when multiple states go into the same measurement, they'll to init at same value)

        # separate init-std-dev for each state, corr = 0
        out_cov = quad_form_diag(std_devs=self.initial_std_dev,
                                 corr_mat=Variable(torch.eye(self.num_states, self.num_states)))

        return expand(out_mean, bs), expand(out_cov, bs)

    # Main Forward-Pass Methods --------------------
    def _filter(self, input, initial_state=None, initializer_input=None, n_ahead=None):
        if n_ahead is None:
            n_ahead = self.forward_ahead

        num_series, num_timesteps, num_measurements = input.data.shape

        # add an extra n_ahead dimension b/c it's needed for k_*_next, but remember this so we can remove it later
        if n_ahead == 0:
            n_ahead_was_zero = True
            n_ahead = 1
        else:
            n_ahead_was_zero = False

        # "preallocate"
        mean_out = [[] for _ in range(n_ahead + 1)]
        cov_out = [[] for _ in range(n_ahead + 1)]

        # initial values:
        if initial_state is None:
            if initializer_input is None:
                # we don't give a warning if not self.use_default_initializer, b/c initializer_input might not need inputs
                initializer_input = Variable(torch.zeros((num_series, num_measurements)))
            elif self.use_default_initializer:
                warn("Passed `initializer_input`, even though no `initializer` was passed at `__init__`.")
            k_mean, k_cov = self.initializer(initializer_input)
        else:
            if initializer_input is None:
                warn("Both `initial_state` and `initializer_input` were passed, ignoring the latter.")
            k_mean, k_cov = initial_state

        # run filter:
        for t in range(-self.forward_ahead, num_timesteps):
            # update
            if t >= 0:
                # the timestep for "initial" depends on self.forward_ahead.
                # e.g., if self.forward_ahead is -1, then we start at -1, and update isn't called yet.
                k_mean, k_cov = self.kf_update(input[:, t, :], k_mean, k_cov)

            # predict
            for nh in range(n_ahead + 1):
                if nh > 0:
                    # if ahead is in the future, need to predict. otherwise will just use posterior for the current time
                    k_mean, k_cov = self.kf_predict(k_mean, k_cov)
                if nh == 1:
                    # if ahead is 1-step ahead, save it for the next iter
                    k_mean_next, k_cov_next = k_mean, k_cov
                if 0 <= (t + nh) < num_timesteps:
                    # don't assign to the matrix unless there's a corresponding observation in the original data
                    mean_out[nh].append(k_mean)
                    cov_out[nh].append(k_cov)

            # next timestep:
            # noinspection PyUnboundLocalVariable
            k_mean, k_cov = k_mean_next, k_cov_next

        # forward-pass is done, so make sure design-mats will be re-instantiated next time:
        self.design.reset()

        # we added an extra n_ahead dimension b/c it was needed for k_*_next. but they didn't ask for that, so remove
        # from final output:
        if n_ahead_was_zero:
            mean_out.pop()
            cov_out.pop()

        return mean_out, cov_out

    def forward(self, input, initial_state=None, initializer_input=None):
        """
        The forward pass.

        :param input: A 3D variable with group * timesteps * X. The first 0:num_measurements slices along the third dimension
        will be passed to the kalman-filter. Any remaining
        :return: A torch.autograd.Variable of size num_groups*num_timesteps*num_measurements, containing the predictions
        for time=self.forward_ahead.
        """

        # filter takes the first num_measurements slices along the third axis. additional slices are allowed and will be used
        # in the measurement_nn
        input = self.validate_input(input)
        kf_input = input[:, :, 0:self.num_measurements]
        num_series, num_timesteps, num_measurements = kf_input.data.shape

        # run kalman-filter to get predicted state
        means_per_ahead, _ = self._filter(input=kf_input,
                                          initial_state=initial_state,
                                          initializer_input=initializer_input,
                                          n_ahead=self.forward_ahead)
        means = torch.cat(means_per_ahead[self.forward_ahead], 0)

        # run through the measurement matrix to get predicted measurements
        H_expanded = expand(self.H, means.data.shape[0])
        filtered_long = torch.bmm(H_expanded, means)
        filtered = filtered_long.view(num_timesteps, num_series, num_measurements).transpose(0, 1)

        # the first num_measurements slices along the third axis were passed though the kalman-filter. if there's a
        # measurement_nn, it will take the entire input
        if self.measurement_nn is None:
            if input.data.shape[2] > kf_input.data.shape[2]:
                warn("There are extra slices along the 3rd dimension of `input` no `measurement_nn` to process them.")
            return filtered
        else:
            measure_predictions = self.measurement_nn(input)
            # the predictions are simply the kalman-filter plus the measurement-nn
            return filtered + measure_predictions

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
    def kf_predict(self, x, P):
        """
        The 'predict' step of the kalman filter. The F matrix dictates how the mean (x) and covariance (P) change.

        :param x: Mean
        :param P: Covariance
        :return: The new (mean, covariance)
        """
        bs = P.data.shape[0]  # batch-size

        # expand design-matrices to match batch-size:
        F_expanded = expand(self.F, bs)
        Ft_expanded = expand(self.F.t(), bs)
        Q_expanded = expand(self.Q, bs)

        x = torch.bmm(F_expanded, x)
        P = torch.bmm(torch.bmm(F_expanded, P), Ft_expanded) + Q_expanded
        return x, P

    def kf_update(self, obs, x, P):
        """
        The 'update' step of the kalman filter.

        :param obs: The observations.
        :param x: Mean
        :param P: Covariance
        :return: The new (mean, covariance)
        """

        # handle missing values
        obs_nm, x_nm, P_nm, is_nan_per_slice = self.nan_remove(obs, x, P)
        if obs_nm is None:  # all missing
            return x, P

        # expand design-matrices to match batch-size:
        bs_orig = obs.data.shape[0]  # batch-size including missings
        bs = obs_nm.data.shape[0]  # batch-size
        H_expanded = expand(self.H, bs)
        R_expanded = expand(self.R, bs)

        # residual:
        residual = obs_nm - torch.bmm(H_expanded, x_nm).squeeze(2)

        # kalman-gain:
        K = self.kalman_gain(P_nm, H_expanded, R_expanded)

        # update mean and covariance:
        x_new = x.clone()
        # from group*one_for_all_states to group*state to group*state*1:
        is_nan_slice_x = is_nan_per_slice[:, None].expand(bs_orig, self.num_states)[:, :, None]
        x_new[is_nan_slice_x == 0] = x_nm + torch.bmm(K, residual.unsqueeze(2))

        P_new = P.clone()
        # group*state*1 to group*state*state:
        is_nan_slice_P = is_nan_slice_x.expand(bs_orig, self.num_states, self.num_states)
        P_new[is_nan_slice_P == 0] = self.covariance_update(P_nm, K, H_expanded, R_expanded)

        return x_new, P_new

    @staticmethod
    def nan_remove(obs, x, P):
        """
        Remove group-slices from x and P where any variables in that slice are missing.

        :param obs: The observed data, potentially with missing values.
        :param x: Mean.
        :param P: Covariance
        :return: A tuple with four elements. The first three are the tensors originally passed as arguments, but without
         any group-slices that have missing values. The fourth is a byte-tensor indicating which slices in the original
         `obs` had missing-values.
        """
        bs = P.data.shape[0]  # batch-size including missings
        rank = P.data.shape[1]
        is_nan_el = (obs != obs)  # by element
        if is_nan_el.data.all():
            return None, None, None, None
        if is_nan_el.data.any():
            # get a list, one for each group, indicating 1 for nan (for *any* variable):
            is_nan_list = (torch.sum(is_nan_el, 1) > 0).data.tolist()
            # keep only the group-slices without nans:
            obs_nm = torch.stack([obs[i] for i, is_nan in enumerate(is_nan_list) if is_nan == 0], 0)
            x_nm = torch.stack([x[i] for i, is_nan in enumerate(is_nan_list) if is_nan == 0], 0)
            P_nm = torch.stack([P[i] for i, is_nan in enumerate(is_nan_list) if is_nan == 0], 0)
        else:
            # don't need to do anything:
            obs_nm, x_nm, P_nm = obs, x, P

        # this will be used later for assigning to x/P _new.
        is_nan_per_slice = (torch.sum(is_nan_el, 1) > 0)

        return obs_nm, x_nm, P_nm, is_nan_per_slice

    @staticmethod
    def kalman_gain(P, H_expanded, R_expanded):
        bs = P.data.shape[0]  # batch-size
        Ht_expanded = expand(H_expanded[0].t(), bs)
        S = torch.bmm(torch.bmm(H_expanded, P), Ht_expanded) + R_expanded  # total covariance
        Sinv = torch.cat([torch.inverse(S[i, :, :]).unsqueeze(0) for i in range(bs)], 0)  # invert, batchwise
        K = torch.bmm(torch.bmm(P, Ht_expanded), Sinv)  # kalman gain
        return K

    @staticmethod
    def covariance_update(P, K, H_expanded, R_expanded):
        """
        "Joseph stabilized" covariance correction.

        :param P: Process covariance.
        :param K: Kalman-gain.
        :param H_expanded: The H design-matrix, expanded for each batch.
        :param R_expanded: The R design-matrix, expanded for each batch.
        :return: The new process covariance.
        """
        rank = H_expanded.data.shape[2]
        I = expand(Variable(torch.eye(rank, rank)), P.data.shape[0])
        p1 = (I - torch.bmm(K, H_expanded))
        p2 = torch.bmm(torch.bmm(p1, P), batch_transpose(p1))
        p3 = torch.bmm(torch.bmm(K, R_expanded), batch_transpose(K))
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
