import torch
from torch.autograd import Variable

from kalman_pytorch.utils.torch_utils import expand, batch_transpose, quad_form_diag


# noinspection PyPep8Naming
class KalmanFilter(torch.nn.Module):
    def __init__(self, forward_ahead=1):
        super(KalmanFilter, self).__init__()
        self.forward_ahead = forward_ahead

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

    @property
    def design(self):
        raise NotImplementedError()

    def initializer(self, tens):
        """
        :param tens: A tensor of observed values.
        :return: Initial values for mean, cov that match the shape of the batch (tens).
        """
        raise NotImplementedError()

    def default_initializer(self, tens, initial_state, initial_std_dev):
        """
        :param tens: A tensor of a batch observed values.
        :param initial_state: A Variable/Parameter that stores the initial state.
        :param initial_std_dev: A Variable/Parameter that stores the initial std-deviation.
        :return: Initial values for mean, cov that match the shape of the batch (tens).
        """
        num_measurements, num_states = self.H.data.shape
        if len(initial_state) == num_states:
            # if the initial state is just as big as the number of states, then mapping is one-to-one
            initial_mean = initial_state[:, None]
        elif len(initial_state) == num_measurements:
            initial_mean = Variable(torch.zeros(num_states, 1))
            for midx in range(num_measurements):
                for sidx,include in enumerate(self.H[midx,:].data.tolist()):
                    if include:
                        initial_mean[sidx] = initial_state[midx] / torch.sum(self.H[midx,:])
        else:
            raise ValueError("The initial state is not a compatible size.")

        initial_cov = quad_form_diag(std_devs=initial_std_dev, corr_mat=Variable(torch.eye(num_states, num_states)))
        bs = tens.data.shape[0]  # batch-size
        return expand(initial_mean, bs), expand(initial_cov, bs)

    # Main Forward-Pass Methods --------------------
    def _filter(self, input, n_ahead):

        num_series, num_timesteps, num_variables = input.data.shape

        if n_ahead == 0:
            n_ahead_was_zero = True
            n_ahead = 1
        else:
            n_ahead_was_zero = False

        # "preallocate"
        mean_out = [[] for _ in range(n_ahead+1)]
        cov_out = [[] for _ in range(n_ahead+1)]

        # initial values:
        k_mean, k_cov = self.initializer(input)

        # run filter:
        for t in range(-self.forward_ahead, num_timesteps):
            # update
            if t >= 0:
                # the timestep for "initial" depends on self.forward_ahead.
                # e.g., if self.forward_ahead is -1, then we start at -1, and update isn't called yet.
                k_mean, k_cov = self.kf_update(input[:, t, :], k_mean, k_cov)

            # predict
            for nh in range(n_ahead+1):
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
            # mean_out = mean_out[:, :, :, :, [0]]
            # cov_out = cov_out[:, :, :, :, [0]]

        return mean_out, cov_out

    def filter(self, input, n_ahead):
        """
        Get the n-step-ahead state mean and covariance.

        :param input: A torch.autograd.Variable of size num_groups*num_timesteps*num_variables.
        :param n_ahead: The number of timesteps ahead predictions are desired for. Each timestep under this will also
        be included (e.g., n_ahead = 7 means the state for time N and the predictions for times N1 through N7).
        :return: The mean and covariance. Each is a num_groups*num_timesteps*num_states*X*n_ahead
        torch.autograd.Variable, where `X` is 1 for the mean and num_states for the covariance.
        """
        input = self.check_input(input)
        mean_out, cov_out = self._filter(input=input, n_ahead=n_ahead)
        # don't forget to pad if n_ahead > self.forward_ahead
        raise NotImplementedError("")

    def predict_ahead(self, input, n_ahead):
        """
        Get the n-step-ahead predictions for measurements -- i.e., the state mapped into measurement-space.

        :param input:  A torch.autograd.Variable of size num_groups*num_timesteps*num_variables.
        :param n_ahead: The number of timesteps ahead predictions are desired for. Each timestep under this will also
        be included (e.g., n_ahead = 7 means the state for time N and the predictions for times N1 through N7).
        :return: A num_groups*num_timesteps*num_variables*n_ahead torch.autograd.Variable, containing the predictions.
        """
        input = self.check_input(input)
        mean_out, _ = self._filter(input=input, n_ahead=n_ahead)
        # don't forget to pad if n_ahead > self.forward_ahead
        raise NotImplementedError("")

    def forward(self, input):
        """
        The forward pass.

        :param input: A torch.autograd.Variable of size num_groups*num_timesteps*num_variables.
        :return: A torch.autograd.Variable of size num_groups*num_timesteps*num_variables, containing the predictions
        for time=self.forward_ahead.
        """
        input = self.check_input(input)
        num_series, num_timesteps, num_variables = input.data.shape

        means_per_ahead, _ = self._filter(input=input, n_ahead=self.forward_ahead)
        means = means_per_ahead[self.forward_ahead]

        # H_expanded = expand(self.H, num_series)
        # out = torch.stack([torch.bmm(H_expanded, k_mean).squeeze(2) for k_mean in means], 1)

        means = torch.cat(means, 0)
        H_expanded = expand(self.H, means.data.shape[0])

        out_long = torch.bmm(H_expanded, means)
        out = out_long.view(num_timesteps, num_series, num_variables).transpose(0, 1)

        return out

    def check_input(self, input):
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
        bs_orig = obs.data.shape[0] # batch-size including missings
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
