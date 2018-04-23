from collections import defaultdict

import torch
from torch.autograd import Variable

from torch_kalman.utils.torch_utils import expand, batch_diag
from numpy import where, nan

from torch_kalman.design import reset_design_on_exit


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

        # when people call design.add_nn_module, if they didn't assign that module to self, they indicate
        # known_to_super=False. when they do that, it's added to this list, and so its guaranteed pytorch will know about it.
        # (useful so that Processes can get added to design and do all the work of adding their nn_modules)
        self.additional_modules = self.design.additional_modules

    @property
    def num_states(self):
        return len(self.design.states)

    @property
    def num_measures(self):
        return len(self.design.measures)

    # Main Forward-Pass Methods --------------------
    @reset_design_on_exit
    def filter(self, initial_state=None, horizon=None, **kwargs):
        """
        Perform kalman-filtering operation on a tensor of data. It's not recommended to call this method directly; instead,
        this method is meant to be called by methods such as `forward`, `components`, and `retrieve_state`, which return
        more organized output than this method.

        :param initial_state: Optional, a tuple containing the initial state mean and initial state covariance. If left as
        None, values for these will be supplied as defined in the `initialize_state` method of the design.
        :param horizon: The forecast horizon.
        :param kwargs: Must include `kf_input`. Other kwargs will be passed to any nn-modules in the design.
        :return: A tuple of dictionaries, this first for the state-mean, the second for the state-cov. The keys are forecast
        horizons (from 0 through `horizon`). The values are themselves dicts, whose keys are timepoints being predicted, and
        the values are the predictions.
        """
        if horizon is None:
            horizon = self.horizon

        num_series, num_timesteps, num_measures = kwargs['kf_input'].data.shape
        if num_measures != self.num_measures:
            raise ValueError("`kf_input`'s size along the third dimension ({}) does not match the number of measures in the "
                             "design ({}).".format(num_measures, self.num_measures))

        # add an extra horizon dimension b/c it's needed for k_*_next, but remember this so we can remove it later
        if horizon == 0:
            horizon_was_zero = True
            horizon = 1
        else:
            horizon_was_zero = False

        # preallocate for clarity
        mean_out, cov_out = {}, {}
        for nh in range(horizon + 1):
            mean_out[nh] = dict.fromkeys(range(num_timesteps))
            cov_out[nh] = dict.fromkeys(range(num_timesteps))

        # initial values:
        if initial_state is None:
            state_mean, state_cov = self.design.initialize_state(**kwargs)
        else:
            state_mean, state_cov = initial_state

        # run filter:
        for t in range(num_timesteps):
            # update: incorporate observed data (and state-nn predictions)
            if t >= 0:
                # if there's a nn module predicting certain components of the state, that is applied here:
                self.design.state_nn_update(state_mean, time=t, **kwargs)
                # now do update step:
                state_mean, state_cov = self.kf_update(state_mean, state_cov, time=t, **kwargs)

            # predict
            for nh in range(horizon + 1):
                t_pred = t + nh
                if t_pred < num_timesteps:  # output only includes timepoints in original input
                    if nh > 0:
                        # if ahead is in the future, need to predict...
                        state_mean, state_cov = self.kf_predict(state_mean, state_cov, time=t_pred, **kwargs)
                    # ...otherwise will just use posterior for the current time

                    if nh == 1:
                        # if ahead is 1-step ahead, save it for the next iter
                        state_mean_next, state_cov_next = state_mean.clone(), state_cov.clone()

                    # output only includes timepoints in original input
                    if t_pred >= 0:
                        mean_out[nh][t_pred] = state_mean
                        cov_out[nh][t_pred] = state_cov

            # next timestep:
            # noinspection PyUnboundLocalVariable
            state_mean, state_cov = state_mean_next, state_cov_next

        # we added an extra horizon dimension b/c it was needed for k_*_next. but they didn't ask for that, so remove
        # from final output:
        if horizon_was_zero:
            mean_out.pop(1)
            cov_out.pop(1)

        return mean_out, cov_out

    @reset_design_on_exit
    def forward(self, kf_input, initial_state=None, **kwargs):
        """
        Perform a forward-pass, with kf_input being a Variable whose first dimension is group, second is time, and third is
        measure. Returns the n-step-ahead predictions for this input, with n=self.horizon. Note that this first self.horizon
        predictions will be nan, so these must be removed before calculating loss.

        :param kf_input: A Variable whose first dimension is group, second is time, and third is measure.
        :param initial_state: Optional, a tuple containing the initial state mean and initial state covariance. If left as
        None, values for these will be supplied as defined in the `initialize_state` method of the design.
        :param kwargs: Kwargs that will be passed to nn-modules in the design.
        :return: A Variable with the n-step-ahead predictions for the input, with n=self.horizon.
        """
        if len(kf_input.data.shape) != 3:
            raise Exception("Unexpected shape for `kf_input`. If your data are univariate, add the singular third dimension "
                            "with kf_input[:,:,None].")
        kwargs['kf_input'] = kf_input

        # run kalman-filter to get predicted state
        means_per_horizon, _ = self.filter(initial_state=initial_state, horizon=self.horizon, **kwargs)
        means = means_per_horizon[self.horizon]

        # get predicted measures from state
        nan_pad = Variable(torch.zeros(kf_input.data.shape[0], self.num_measures, 1))
        nan_pad[:, :, :] = nan
        predicted_measures = []
        for t, mean in means.items():
            if mean is None:
                predicted_measures.append(nan_pad)
            else:
                H_expanded = self.H.create_for_batch(time=t, **kwargs)
                predicted_measures.append(torch.bmm(H_expanded, mean))

        return torch.cat(predicted_measures, 2).permute(0, 2, 1)

    @reset_design_on_exit
    def components(self, kf_input, initial_state=None, horizon=None, **kwargs):
        """
        Return the components of the kalman-filter's state. This is helpful for understanding what goes into predictions/
        forecasts.

        :param kf_input: The main input to the kalman-filter. A Variable whose first dimension is group, second is time, and
        third is measure.
        :param initial_state: Optional, a tuple containing the initial state mean and initial state covariance. If left as
        None, values for these will be supplied as defined in the `initialize_state` method of the design.
        :param horizon: The forecast-horizon.
        :param kwargs: Kwargs that will be passed to nn-modules in the design.
        :return: A dictionary with the names of the observable states as keys and the state-means as values. The state-means
        will have dimensions group X time X 1.
        """
        if len(kf_input.data.shape) != 3:
            raise Exception("Unexpected shape for `kf_input`. If your data are univariate, add the singular third dimension "
                            "with kf_input[:,:,None].")
        kwargs['kf_input'] = kf_input

        # run kalman-filter to get predicted state
        horizon = horizon or self.horizon
        means_per_horizon, _ = self.filter(initial_state=initial_state, horizon=horizon, **kwargs)
        means = means_per_horizon[horizon]

        # get the observable states:
        nan_pad = Variable(torch.zeros(kf_input.data.shape[0], 1, 1))
        nan_pad[:, :, :] = nan
        observable_states = {state.id: self.design.state_idx[state.id] for state in self.design.measurable_states}
        components = defaultdict(list)
        for t, mean in means.items():
            if mean is None:
                for state_id in observable_states.keys():
                    components[state_id].append(nan_pad)
            else:
                for state_id, idx in observable_states.items():
                    components[state_id].append(mean[:, [idx], :])

        return {name: torch.cat(component, 1) for name, component in components.items()}

    @reset_design_on_exit
    def forecast(self, initial_state, horizon=None, **kwargs):
        """
        Generate a forecast given an initial state.

        :param initial_state: The initial state. To generate from past data, use the `retrieve_state` method.
        :param horizon: The forecast horizon.
        :param kwargs: Kwargs that will be passed to nn-modules in the design.
        :return: (Forecast means , forecast std-devs). Each is a Variable with group X time X measure.
        """
        if horizon is None:
            horizon = self.horizon

        state_mean, state_cov = initial_state

        # kf_input is used by methods inside of kf_predict to get the batch size, so send a placeholder
        bs = state_mean.data.shape[0]
        kf_input = torch.zeros(bs, horizon, self.num_measures)
        kf_input[:, :, :] = nan
        kwargs['kf_input'] = Variable(kf_input)

        out_means, out_std_devs = [], []
        for t in range(horizon):
            # generate state for this timestep:
            self.design.state_nn_update(state_mean, time=t, **kwargs)
            state_mean, state_cov = self.kf_predict(state_mean, state_cov, time=t, **kwargs)

            # translate into observable measures:
            H_expanded = self.H.create_for_batch(time=t, **kwargs)
            Ht_expanded = H_expanded.permute(0, 2, 1)
            pred_mean = torch.bmm(H_expanded, state_mean).squeeze()
            pred_cov = torch.bmm(torch.bmm(H_expanded, state_cov), Ht_expanded)
            pred_std_dev = torch.sqrt(batch_diag(pred_cov))
            out_means.append(pred_mean[:, None, :])
            out_std_devs.append(pred_std_dev[:, None, :])

        return torch.cat(out_means, 1), torch.cat(out_std_devs, 1)

    @reset_design_on_exit
    def retrieve_state(self, kf_input, times=None, initial_state=None, **kwargs):
        """
        Get the state mean and covariance for a particular dataset, optionally at a different timeslice for each group.

        :param kf_input: The main input to the kalman-filter. A Variable whose first dimension is group, second is time, and
        third is measure.
        :param times: Optional. An iterable of times at which to extract the state, one time per group in `kf_input`. If not
        supplied then will use the final time for all groups (this is typically not what's wanted for time-series of variable
         length, since the short series are padded with nans).
        :param initial_state: Optional, a tuple containing the initial state mean and initial state covariance. If left as
        None, values for these will be supplied as defined in the `initialize_state` method of the design.
        :param kwargs: Kwargs that will be passed to nn-modules in the design.
        :return: The state mean, state covariance.
        """
        if len(kf_input.data.shape) != 3:
            raise Exception("Unexpected shape for `kf_input`. If your data are univariate, add the singular third dimension "
                            "with kf_input[:,:,None].")
        kwargs['kf_input'] = kf_input

        means_per_horizon, covs_per_horizon = self.filter(initial_state=initial_state, horizon=0, **kwargs)
        means = means_per_horizon[0]
        covs = covs_per_horizon[0]

        if times is None:
            return means[len(means) - 1], covs[len(covs) - 1]

        out_means = []
        out_covs = []
        for g, time in enumerate(times):
            out_means.append(means[time][[g], :, :])
            out_covs.append(covs[time][[g], :, :])

        return torch.cat(out_means, 0), torch.cat(out_covs, 0)

    # Kalman-Smoother ------------------------------
    @reset_design_on_exit
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
            if not this_isnan.all():  # if all nan, just don't perform update
                # for partial nan, perform partial update
                valid_idx = where(this_isnan.numpy() == 0)[0].tolist()  # will clean up when pytorch 0.4 is released
                # get the subset of measures that are non-nan:
                K_sub = K[i][:, valid_idx]
                residual_sub = residual[i][valid_idx]
                H_sub = H_expanded[i][valid_idx, :]
                R_sub = R_expanded[i][valid_idx][:, valid_idx]  # will clean up when pytorch 0.4 is released

                # perform updates with subsetted matrices
                state_mean_new[i] = state_mean[i] + torch.mm(K_sub, residual_sub.unsqueeze(1))
                state_cov_new[i] = \
                    self.covariance_update(state_cov[[i]], K_sub[None, :, :], H_sub[None, :, :], R_sub[None, :, :])[0]

        no_nan = [i for i in range(bs) if i not in groups_with_nan]
        if len(no_nan) > 0:
            state_mean_new[no_nan] = state_mean[no_nan] + torch.bmm(K[no_nan], residual[no_nan].unsqueeze(2))
            state_cov_new[no_nan] = self.covariance_update(state_cov[no_nan], K[no_nan], H_expanded[no_nan],
                                                           R_expanded[no_nan])

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
