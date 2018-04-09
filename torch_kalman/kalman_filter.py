import torch
from torch.autograd import Variable

from torch_kalman.utils.torch_utils import expand
from numpy import where


# noinspection PyPep8Naming
class KalmanFilter(torch.nn.Module):
    def __init__(self, design, horizon=1):
        """
        :param horizon: An integer indicating how many timesteps ahead predictions from `forward` should be for.
        """
        super(KalmanFilter, self).__init__()
        self.horizon = horizon
        self.design = design
        if self.design.finalized:
            raise Exception("Please pass the design to KalmanFilter *before* finalizing it, then finalize it in your "
                            "child-module's __init__ method.")

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
    def _filter(self, initial_state=None, n_ahead=None, **kwargs):
        if n_ahead is None:
            n_ahead = self.horizon

        num_series, num_timesteps, num_measurements = kwargs['kf_input'].data.shape

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
        if initial_state is None:
            state_mean, state_cov = self.design.initialize_state(**kwargs)
        else:
            state_mean, state_cov = initial_state

        # run filter:
        for t in range(-self.horizon, num_timesteps):
            # update: incorporate observed data (and state-nn predictions)
            if t >= 0:
                # if there's a nn module predicting certain components of the state, that is applied here:
                self.design.state_nn_update(state_mean, time=t, **kwargs)
                # now do update step:
                state_mean, state_cov = self.kf_update(state_mean, state_cov, time=t, **kwargs)

            # predict
            for nh in range(n_ahead + 1):
                assign_idx = t + nh
                if assign_idx < num_timesteps:  # output only includes timepoints in original input
                    if nh > 0:
                        # if ahead is in the future, need to predict...
                        state_mean, state_cov = self.kf_predict(state_mean, state_cov, time=assign_idx, **kwargs)
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

    def forward(self, initial_state=None, **kwargs):
        if len(kwargs['kf_input'].data.shape) != 3:
            raise Exception("Unexpected shape for `kf_input`. If your data are univariate, add the singular third dimension "
                            "with kf_input[:,:,None].")

        # run kalman-filter to get predicted state
        means_per_ahead, _ = self._filter(initial_state=initial_state, n_ahead=self.horizon, **kwargs)
        means = means_per_ahead[self.horizon]

        # get predicted measurements from state
        predicted_measurements = []
        for t, mean in enumerate(means):
            H_expanded = self.H.create_for_batch(time=t, **kwargs)
            predicted_measurements.append(torch.bmm(H_expanded, mean))

        return torch.cat(predicted_measurements, 2).permute(0, 2, 1)

    # Kalman-Smoother ------------------------------
    def smooth(self, input):
        if input is not None:
            raise NotImplementedError("Kalman-smoothing is not yet implemented.")
        return None

    # Main KF Steps -------------------------
    def kf_predict(self, state_mean, state_cov, time, **kwargs):
        """
        The 'predict' step of the kalman filter.

        :param obs: The observed data for this timestep. Not needed for the predict computations themselves, but will be
        passed to the `create_for_batch` methods of the design-matrices.
        :param state_mean: The state mean (for each item in the batch).
        :param state_cov: The state covariance (for each item in the batch).
        :return: The update state_mean, state_cov.
        """
        # expand design-matrices for the batch
        F_expanded = self.F.create_for_batch(time=time, **kwargs)
        Ft_expanded = F_expanded.permute(0, 2, 1)
        Q_expanded = self.Q.create_for_batch(time=time, **kwargs)

        state_mean = torch.bmm(F_expanded, state_mean)
        state_cov = torch.bmm(torch.bmm(F_expanded, state_cov), Ft_expanded) + Q_expanded
        return state_mean, state_cov

    def kf_update(self, state_mean, state_cov, time, **kwargs):

        # expand design-matrices for the batch:
        H_expanded = self.H.create_for_batch(time=time, **kwargs)
        R_expanded = self.R.create_for_batch(time=time, **kwargs)

        # for the actual update, we only use the autoregressive part of the input:
        kf_obs = kwargs['kf_input'][:, time, :]

        # residual:
        residual = kf_obs - torch.bmm(H_expanded, state_mean).squeeze(2)

        # kalman-gain:
        K = self.kalman_gain(state_cov, H_expanded, R_expanded)

        # handle missing values:
        state_mean_new, state_cov_new = state_mean.clone(), state_cov.clone()  # avoid 'var needed for grad modified inplace'
        isnan = (residual != residual)
        bs = kf_obs.data.shape[0]
        groups_with_nan = [i for i in range(bs) if isnan[i].data.any()]
        for i in groups_with_nan:
            this_isnan = isnan[i].data
            if this_isnan.all():
                # if all nan, just don't perform update
                continue
            else:
                # for partial nan, perform partial update
                valid_idx = where(this_isnan.numpy() == 0)[0].tolist()  # will clean up when pytorch 0.4 is released
                # get the subset of measurements that are non-nan:
                K_sub = K[i][:, valid_idx]
                residual_sub = residual[i][valid_idx]
                H_sub = H_expanded[i][valid_idx, :]
                R_sub = R_expanded[i][valid_idx][:, valid_idx]  # will clean up when pytorch 0.4 is released

                # perform updates with subsetted matrices
                state_mean_new[i] = state_mean[i] + torch.mm(K_sub, residual_sub.unsqueeze(1))
                state_cov_new[i] = self.covariance_update(state_cov[[i]], K_sub[None, :, :], H_sub[None, :, :], R_sub[None, :, :])[0]

        no_nan = [i for i in range(bs) if i not in groups_with_nan]
        if len(no_nan) > 0:
            state_mean_new[no_nan] = state_mean[no_nan] + torch.bmm(K[no_nan], residual[no_nan].unsqueeze(2))
            state_cov_new[no_nan] = self.covariance_update(state_cov[no_nan], K[no_nan], H_expanded[no_nan], R_expanded[no_nan])

        return state_mean_new, state_cov_new

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
        bs = state_cov.data.shape[0]  # batch size
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
