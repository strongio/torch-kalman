import torch
from torch.autograd import Variable

from kalman_pytorch.utils.torch_utils import expand, batch_transpose, quad_form_diag


# noinspection PyPep8Naming
class KalmanFilter(torch.nn.Module):
    def __init__(self):
        super(KalmanFilter, self).__init__()

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
            # TODO: this code doesn't work
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
    def predict_ahead(self, x, n_ahead):
        """
        Given a time-series X -- a 3D tensor with group * variable * time -- generate predictions -- a 4D tensor with
        group * variable * time * n_ahead.

        :param x: A 3D tensor with group * variable * time.
        :param n_ahead: The number of steps ahead for prediction. Minumum is 1.
        :return: Predictions: a 4D tensor with group * variable * time * n_ahead.
        """

        if not isinstance(x.data, torch.FloatTensor):
            raise ValueError("The data for `x` must be a torch.FloatTensor.")

        # data shape:
        if len(x.data.shape) != 3:
            raise Exception("`x` should have three-dimensions: group*variable*time. "
                            "If there's only one dimension and current structure is group*time, "
                            "reshape with x[:,None,:].")
        num_series, num_variables, num_timesteps = x.data.shape

        # initial values:
        k_mean, k_cov = self.initializer(x)

        # preallocate:
        output = Variable(torch.zeros(list(x.data.shape) + [n_ahead]))

        # fill one timestep at a time
        for i in xrange(num_timesteps):
            # predict n-ahead
            for nh in xrange(n_ahead):
                k_mean, k_cov = self.kf_predict(k_mean, k_cov)
                if nh == 0:
                    k_mean_next, k_cov_next = k_mean, k_cov
                output[:, :, i, nh] = torch.bmm(expand(self.H, num_series), k_mean)

            # but for next timestep, only use 1-ahead:
            # noinspection PyUnboundLocalVariable
            k_mean, k_cov = k_mean_next, k_cov_next

            # update. note `[i]` instead of `i` (keeps dimensionality)
            k_mean, k_cov = self.kf_update(x[:, :, [i]], k_mean, k_cov)

        # forward-pass is done, so make sure design-mats will be re-instantiated next time:
        self.design.reset()

        #
        return output

    def forward(self, x):
        """
        The forward pass.

        :param x: A 3D tensor with group * variable * time
        :return: A 3D tensor with model output.
        """
        out = self.predict_ahead(x, n_ahead=1)
        return torch.squeeze(out, 3)  # flatten

    # Kalman-Smoother ------------------------------
    def smooth(self, x):
        if x is not None:
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

        :param obs: The observations. (TODO: does it need to be 3D?)
        :param x: Mean
        :param P: Covariance
        :return: The new (mean, covariance)
        """

        # handle missing values
        obs_nm, x_nm, P_nm, is_nan_slice = self.nan_remove(obs, x, P)
        if obs_nm is None: # all missing
            return x, P

        # expand design-matrices to match batch-size:
        bs = P_nm.data.shape[0]  # batch-size
        H_expanded = expand(self.H, bs)
        R_expanded = expand(self.R, bs)

        # residual:
        residual = obs_nm - torch.bmm(H_expanded, x_nm)

        # kalman-gain:
        K = self.kalman_gain(P_nm, H_expanded, R_expanded)

        # update mean and covariance:
        x_new, P_new = x.clone(), P.clone()
        x_new[is_nan_slice == 0] = x_nm + torch.bmm(K, residual)
        P_new[is_nan_slice == 0] = self.covariance_update(P_nm, K, H_expanded, R_expanded)

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
            # get a list, one for each group-slice, indicating 1 for nan:
            is_nan_list = (torch.sum(is_nan_el, 1) > 0).squeeze(1).data.tolist()
            # keep only the group-slices without nans:
            obs_nm = torch.stack([obs[i] for i, is_nan in enumerate(is_nan_list) if is_nan == 0], 0)
            x_nm = torch.stack([x[i] for i, is_nan in enumerate(is_nan_list) if is_nan == 0], 0)
            P_nm = torch.stack([P[i] for i, is_nan in enumerate(is_nan_list) if is_nan == 0], 0)
        else:
            # don't need to do anything:
            obs_nm, x_nm, P_nm = obs, x, P

        # this is used to index into the original x,P when performing assignment later:
        is_nan_slice = (torch.sum(is_nan_el, 1, keepdim=True) > 0).expand(bs, rank, 1)

        return obs_nm, x_nm, P_nm, is_nan_slice

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
