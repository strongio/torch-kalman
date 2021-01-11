from collections import defaultdict
from typing import Tuple, Union, Sequence, Optional, Dict
from warnings import warn

import torch
from torch import nn, Tensor

import numpy as np

from torch_kalman.utils.data import TimeSeriesDataset

Selector = Union[int, slice, Sequence[int]]


class StateBeliefOverTime(nn.Module):
    """
    The output of the KalmanFilter forward pass.

    TODO: better name?

    Contains methods for evaluating the predictions (log_prob), converting them into dataframes (to_dataframe), and for
    sampling from the underlying distribution (sample_measurements).
    """

    def __init__(self,
                 means: Tensor,
                 covs: Tensor,
                 R: Tensor,
                 H: Tensor,
                 kf_step: 'KFStep'):
        super().__init__()
        self.means = means
        self.covs = covs
        self.H = H
        self.R = R

        self.kf_step = kf_step

        self._predictions = None
        self._prediction_uncertainty = None

        self.num_groups, self.num_timesteps, self.state_size = self.means.shape

    def forward(self, obs: Tensor) -> Tensor:
        """
        Compute the log-probability of data (e.g. data that was originally fed into the KalmanFilter).

        :param obs: A Tensor that could be used in the KalmanFilter.forward pass.
        :param kwargs: Other keyword arguments needed to evaluate the log-prob (e.g. for a censored-kalman-filter, the
          upper and lower bounds).
        :return: A tensor with one element for each group X timestep indicating the log-probability.
        """
        num_groups, num_times, num_dist_dims = obs.shape
        assert self.predictions.shape[2] == num_dist_dims

        """
        group into chunks for log-prob evaluation. the way indexing works makes this tricky, and slow if we just create 
        a separate group X measure index for each separate time-slice. two shortcuts are used to mitigate this:
        (1) the first N time-slices that are nan-free will all be evaluated as a chunk
        (2) subsequent nan-free slices use `slice` notation instead of having to iterate through each group, checking
            which measures were nan
        For all other time-points, we need a separate (group-indices, time-index, measure-indices) tuple.
        """

        times_without_nan = list()
        last_nonan_t = -1
        lp_groups = defaultdict(list)
        for t in range(num_times):
            if torch.isnan(obs[:, t]).all():
                # no log-prob needed
                continue

            if not torch.isnan(obs[:, t]).any():
                # will be updated as block:
                if last_nonan_t == (t - 1):
                    last_nonan_t += 1
                else:
                    times_without_nan.append(t)
                continue

            for g in range(num_groups):
                is_nan = torch.isnan(obs[g, t])
                if is_nan.all():
                    # no log-prob needed
                    continue
                measure_idx = self._which_valid_key(is_nan)
                lp_groups[(t, measure_idx)].append(g)

        lp_groups = [(gidx, t, midx) for (t, midx), gidx in lp_groups.items()]

        # shortcuts:
        if last_nonan_t >= 0:
            gtm = slice(None), slice(last_nonan_t + 1), slice(None)
            lp_groups.append(gtm)
        if len(times_without_nan):
            gtm = slice(None), times_without_nan, slice(None)
            lp_groups.append(gtm)

        # compute log-probs by dims available:
        out = torch.zeros((num_groups, num_times))
        for group_idx, time_idx, measure_idx in lp_groups:
            if isinstance(time_idx, int):
                # assignment is dimensionless in time; needed b/c group isn't a slice
                out[group_idx, time_idx] = self.kf_step.likelihood(
                    obs=obs,
                    group_idx=group_idx,
                    time_idx=(time_idx,),
                    measure_idx=measure_idx
                ).squeeze(-1)
            else:
                # time has dimension, but group is a slice so it's OK
                out[group_idx, time_idx] = self.kf_step.likelihood(
                    obs=obs,
                    group_idx=group_idx,
                    time_idx=time_idx,
                    measure_idx=measure_idx
                )

        return out

    # Information for Prediction ---------:
    @property
    def predictions(self) -> Tensor:
        """
        The predictions on the measurement scale -- i.e., appropriate for comparing to the input that was originally fed
         into the KalmanFilter, e.g. via metrics like MSE.
        """
        if self._predictions is None:
            self._predictions = self.H.matmul(self.means.unsqueeze(-1)).squeeze(-1)
        return self._predictions

    @property
    def prediction_uncertainty(self) -> Tensor:
        """
        Uncertainty on the measurement scale, aka "system uncertainty".
        """
        if self._prediction_uncertainty is None:
            Ht = self.H.permute(0, 1, 3, 2)
            cov = self.H.matmul(self.covs).matmul(Ht) + self.R
            if (cov < 0).any():
                warn(
                    f"negative values in `prediction_uncertainty`. This can be caused by "
                    f"`{type(self).__name__}().covs` not being positive-definite. Try stepping through each group,time "
                    f"of this matrix to find the offending matrix (e.g. torch.cholesky returns an error); then inspect "
                    f"the observations around this group/time."
                )
        return self._prediction_uncertainty

    def get_timeslice(self, indices: Dict[int, Selector]) -> 'StateBeliefOverTime':
        """
        For each group, get a time or slice of times. Often useful because means,covs 0-timestep will be the first dt
        that each group data data, and all groups will be right-padded to be equal in length; but in some forecasting
        contexts we want to take the *last* timestep for each group.

        :param indices: A dictionary where the keys are group-indices and the values are for taking the time-slice
        (e.g. an integer to get the specific timepoint, or a slice to get multiple timepoints).
        :return: A new `StateBeliefOverTime`
        """
        if isinstance(indices, dict):
            indices = list(indices.items())
        means, covs, H, R = [], [], [], []
        for g, t in indices:
            means.append(self.means[g, t])
            covs.append(self.covs[g, t])
            H.append(self.H[g, t])
            R.append(self.R[g, t])
        return type(self)(means.stack(0), covs.stack(0), R=R.stack(0), H=H.stack(0), kf_step=self.kf_step)

    # Exporting to other Formats ---------:
    def to_dataframe(self,
                     dataset: Union[TimeSeriesDataset, dict],
                     type: str = 'predictions',
                     group_colname: str = 'group',
                     time_colname: str = 'time',
                     multi: Optional[float] = 1.96) -> 'DataFrame':
        """
        :param dataset: Either a TimeSeriesDataset, or a dictionary with 'start_times', 'group_names', & 'dt_unit'
        :param type: Either 'predictions' or 'components'.
        :param group_colname: Column-name for 'group'
        :param time_colname: Column-name for 'time'
        :param multi: Multiplier on std-dev for lower/upper CIs. Default 1.96.
        :return: A pandas DataFrame with group, 'time', 'measure', 'mean', 'lower', 'upper'. For type='components'
        additionally includes: 'process' and 'state_element'.
        """

        from pandas import concat

        if isinstance(dataset, TimeSeriesDataset):
            batch_info = {
                'start_times': dataset.start_times,
                'group_names': dataset.group_names,
                'named_tensors': {},
                'dt_unit': dataset.dt_unit
            }
            for measure_group, tensor in zip(dataset.measures, dataset.tensors):
                for i, measure in enumerate(measure_group):
                    if measure in self.design.measures:
                        batch_info['named_tensors'][measure] = tensor[..., [i]]
            missing = set(self.design.measures) - set(dataset.all_measures)
            if missing:
                raise ValueError(
                    f"Some measures in the design aren't in the dataset.\n"
                    f"Design: {missing}\nDataset: {dataset.all_measures}"
                )
        elif isinstance(dataset, dict):
            batch_info = dataset
        else:
            raise TypeError(
                "Expected `batch` to be a TimeSeriesDataset, or a dictionary with 'start_times' and 'group_names'."
            )

        dt_helper = None  # TODO

        def _tensor_to_df(tens, measures):
            times = dt_helper.make_grid(batch_info['start_times'], tens.shape[1])
            return TimeSeriesDataset.tensor_to_dataframe(
                tensor=tens,
                times=times,
                group_names=batch_info['group_names'],
                group_colname=group_colname,
                time_colname=time_colname,
                measures=measures
            )

        assert group_colname not in {'mean', 'lower', 'upper', 'std'}
        assert time_colname not in {'mean', 'lower', 'upper', 'std'}

        out = []
        if type == 'predictions':

            stds = torch.diagonal(self.prediction_uncertainty, dim1=-1, dim2=-2).sqrt()
            for i, measure in enumerate(self.design.measures):
                # predicted:
                df = _tensor_to_df(torch.stack([self.predictions[..., i], stds[..., i]], 2), measures=['mean', 'std'])
                if multi is not None:
                    df['lower'] = df['mean'] - multi * df['std']
                    df['upper'] = df['mean'] + multi * df.pop('std')

                # actual:
                orig_tensor = batch_info.get('named_tensors', {}).get(measure, None)
                if orig_tensor is not None and (orig_tensor == orig_tensor).any():
                    df_actual = _tensor_to_df(orig_tensor, measures=['actual'])
                    df = df.merge(df_actual, on=[group_colname, time_colname], how='left')

                out.append(df.assign(measure=measure))

        elif type == 'components':
            # components:
            for (measure, process, state_element), (m, std) in self._components().items():
                df = _tensor_to_df(torch.stack([m, std], 2), measures=['mean', 'std'])
                if multi is not None:
                    df['lower'] = df['mean'] - multi * df['std']
                    df['upper'] = df['mean'] + multi * df.pop('std')
                df['process'], df['state_element'], df['measure'] = process, state_element, measure
                out.append(df)

            # residuals:
            named_tensors = batch_info.get('named_tensors', {})
            for i, measure in enumerate(self.design.measures):
                orig_tensor = named_tensors.get(measure)
                predictions = self.predictions[..., [i]]
                if orig_tensor.shape[1] < predictions.shape[1]:
                    orig_aligned = predictions.data.clone()
                    orig_aligned[:] = float('nan')
                    orig_aligned[:, 0:orig_tensor.shape[1], :] = orig_tensor
                else:
                    orig_aligned = orig_tensor[:, 0:predictions.shape[1], :]

                df = _tensor_to_df(predictions - orig_aligned, ['mean'])
                df['process'], df['state_element'], df['measure'] = 'residuals', 'residuals', measure
                out.append(df)

        else:
            raise ValueError("Expected `type` to be 'predictions' or 'components'.")

        return concat(out, sort=True)

    def _components(self) -> Dict[Tuple[str, str, str], Tuple[Tensor, Tensor]]:
        states_per_measure = defaultdict(list)
        for state_belief in self.state_beliefs:
            for m, measure in enumerate(self.design.measures):
                H = state_belief.H[:, m, :].data
                m = H * state_belief.means.data
                std = H * torch.diagonal(state_belief.covs.data, dim1=-2, dim2=-1).sqrt()
                states_per_measure[measure].append((m, std))

        out = {}
        for measure, means_and_stds in states_per_measure.items():
            means, stds = zip(*means_and_stds)
            means = torch.stack(means).permute(1, 0, 2)
            stds = torch.stack(stds).permute(1, 0, 2)
            for s, (process_name, state_element) in enumerate(self.design.state_elements):
                if ~torch.isclose(means[:, :, s].abs().max(), torch.zeros(1)):
                    out[(measure, process_name, state_element)] = (means[:, :, s], stds[:, :, s])
        return out

    @staticmethod
    def plot(df: 'DataFrame',
             group_colname: str = None,
             time_colname: str = None,
             max_num_groups: int = 1,
             split_dt: Optional[np.datetime64] = None,
             **kwargs) -> 'DataFrame':
        """
        :param df: The output of `.to_dataframe()`.
        :param group_colname: The name of the group-column.
        :param time_colname: The name of the time-column.
        :param max_num_groups: Max. number of groups to plot; if the number of groups in the dataframe is greater than
        this, a random subset will be taken.
        :param split_dt: If supplied, will draw a vertical line at this date (useful for showing pre/post validation).
        :param kwargs: Further keyword arguments to pass to `plotnine.theme` (e.g. `figure_size=(x,y)`)
        :return: A plot of the predicted and actual values.
        """

        from plotnine import (
            ggplot, aes, geom_line, geom_ribbon, facet_grid, facet_wrap, theme_bw, theme, ylab, geom_vline
        )

        is_components = ('process' in df.columns and 'state_element' in df.columns)

        if group_colname is None:
            group_colname = 'group'
            if group_colname not in df.columns:
                raise TypeError("Please specify group_colname")
        if time_colname is None:
            time_colname = 'time'
            if 'time' not in df.columns:
                raise TypeError("Please specify time_colname")

        df = df.copy()
        if 'upper' not in df.columns and 'std' in df.columns:
            df['upper'] = df['mean'] + 1.96 * df['std']
            df['lower'] = df['lower'] - 1.96 * df['std']
        if df[group_colname].nunique() > max_num_groups:
            subset_groups = df[group_colname].drop_duplicates().sample(max_num_groups).tolist()
            if len(subset_groups) < df[group_colname].nunique():
                print("Subsetting to groups: {}".format(subset_groups))
            df = df.loc[df[group_colname].isin(subset_groups), :]
        num_groups = df[group_colname].nunique()

        aes_kwargs = {'x': time_colname}
        if is_components:
            aes_kwargs['group'] = 'state_element'

        plot = (
                ggplot(df, aes(**aes_kwargs)) +
                geom_line(aes(y='mean'), color='#4C6FE7', size=1.5, alpha=.75) +
                geom_ribbon(aes(ymin='lower', ymax='upper'), color=None, alpha=.25) +
                ylab("")
        )

        if is_components:
            num_processes = df['process'].nunique()
            if num_groups > 1 and num_processes > 1:
                raise ValueError("Cannot plot components for > 1 group and > 1 processes.")
            elif num_groups == 1:
                plot = plot + facet_wrap(f"~ measure + process", scales='free_y', labeller='label_both')
                if 'figure_size' not in kwargs:
                    from plotnine.facets.facet_wrap import n2mfrow
                    nrow, _ = n2mfrow(len(df[['process', 'measure']].drop_duplicates().index))
                    kwargs['figure_size'] = (12, nrow * 2.5)
            else:
                plot = plot + facet_grid(f"{group_colname} ~ measure", scales='free_y', labeller='label_both')
                if 'figure_size' not in kwargs:
                    kwargs['figure_size'] = (12, num_groups * 2.5)

            if (df.groupby('measure')['process'].nunique() <= 1).all():
                plot = plot + geom_line(aes(y='mean', color='state_element'), size=1.5)

        else:
            if 'actual' in df.columns:
                plot = plot + geom_line(aes(y='actual'))
            if num_groups > 1:
                plot = plot + facet_grid(f"{group_colname} ~ measure", scales='free_y', labeller='label_both')
            else:
                plot = plot + facet_wrap("~measure", scales='free_y', labeller='label_both')

            if 'figure_size' not in kwargs:
                kwargs['figure_size'] = (12, 5)

        if split_dt:
            plot = plot + geom_vline(xintercept=np.datetime64(split_dt), linetype='dashed')

        return plot + theme_bw() + theme(**kwargs)
