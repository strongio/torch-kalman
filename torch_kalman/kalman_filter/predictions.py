from collections import defaultdict
from typing import Tuple, Union, Optional, Dict, Iterator
from warnings import warn

import torch
from torch import nn, Tensor

import numpy as np

from torch_kalman.utils.data import TimeSeriesDataset


class Predictions(nn.Module):
    """
    The output of the KalmanFilter forward pass.

    Contains methods for evaluating the predictions (log_prob), converting them into dataframes (to_dataframe), and for
    sampling from the underlying distribution (sample_measurements).
    """

    def __init__(self,
                 state_means: Tensor,
                 state_covs: Tensor,
                 R: Tensor,
                 H: Tensor,
                 kalman_filter: 'KalmanFilter'):
        super().__init__()
        self.state_means = state_means
        self.state_covs = state_covs
        self.H = H
        self.R = R

        self.kalman_filter = kalman_filter

        self._means = None
        self._covs = None

        self.num_groups, self.num_timesteps, self.state_size = self.state_means.shape

    def forward(self, *args, **kwargs):
        return self.forecast(*args, **kwargs)

    def forecast(self,
                 horizon: int,
                 **kwargs):
        start_datetimes = kwargs.get('start_datetimes', None)
        forecast_datetimes = kwargs.get('forecast_datetimes', None)
        if start_datetimes is None:
            if forecast_datetimes is not None:
                raise
            initial_state = self.get_last_update()
        else:
            initial_state = self.get_timeslice(forecast_datetimes, start_datetimes)
        return self.kalman_filter(
            input=None,
            out_timesteps=horizon,
            initial_state=initial_state,
            **kwargs
        )

    def get_last_update(self) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError  # TODO

    def get_timeslice(self, time_idx: np.ndarray, start_times: Optional[np.ndarray] = None) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError  # TODO

    def __iter__(self) -> Iterator[Tensor]:
        # for mean, cov = tuple(predictions)
        yield self.means
        yield self.covs

    def __array__(self) -> np.ndarray:
        # for numpy.asarray
        return self.means.detach().numpy()

    def __getitem__(self, item: Tuple) -> 'Predictions':
        kwargs = {
            'state_means': self.state_means[item],
            'state_covs': self.state_covs[item],
            'H': self.H[item],
            'R': self.R[item]
        }
        cls = type(self)
        for k, v in kwargs.items():
            if k == 'state_means':
                if len(v.shape) != 3:
                    raise TypeError(
                        f"Indexing/slicing into a `{cls.__name__}` object should be done in a way that preserves its "
                        f"3/4D shape; but `means` got shape `{v.shape}`."
                    )
            elif v.shape[-1] != v.shape[-2]:
                # handle symmetry
                if isinstance(item, tuple):
                    if item[0] is Ellipsis and len(item) == 2:
                        # `(..., idx)`, so missed 3rd dim
                        kwargs[k] = v[..., item[1], :]
                        continue
                    elif len(item) == 3:
                        # `(g, t, idx)` is implicitly `(g, t, idx, :)`, so missed 4th dim
                        kwargs[k] = v[..., item[2]]
                        continue
                raise TypeError(
                    f"Indexing/slicing into a `{cls.__name__}` object resulted in symetrical matrix `{k}` "
                    f"having shape {v.shape}. Try `predictions[...,idx]`."
                )
        kwargs['kalman_filter'] = self.kalman_filter
        return cls(**kwargs)

    @property
    def means(self) -> Tensor:
        if self._means is None:
            self._means = self.H.matmul(self.state_means.unsqueeze(-1)).squeeze(-1)
        return self._means

    @property
    def covs(self) -> Tensor:
        if self._covs is None:
            Ht = self.H.permute(0, 1, 3, 2)
            self._covs = self.H.matmul(self.state_covs).matmul(Ht) + self.R
            if (self._covs.diagonal(dim1=-2, dim2=-1) < 0).any():
                warn(
                    f"Negative variance. This can be caused by "
                    f"`{type(self).__name__}().covs` not being positive-definite. Try stepping through each (group,time) "
                    f"of this matrix to find the offending matrix (e.g. torch.cholesky returns an error); then inspect "
                    f"the observations around this group/time."
                )
        return self._covs

    def log_prob(self, obs: Tensor) -> Tensor:
        """
        Compute the log-probability of data (e.g. data that was originally fed into the KalmanFilter).

        :param obs: A Tensor that could be used in the KalmanFilter.forward pass.
        :param kwargs: Other keyword arguments needed to evaluate the log-prob (e.g. for a censored-kalman-filter, the
          upper and lower bounds).
        :return: A tensor with one element for each group X timestep indicating the log-probability.
        """
        assert len(obs.shape) == 3
        assert obs.shape[-1] == self.means.shape[-1]
        ndim = obs.shape[-1]

        obs_flat = obs.view(-1, ndim)
        means_flat = self.means.view(-1, ndim)
        covs_flat = self.covs.view(-1, ndim, ndim)

        lp_flat = torch.zeros(obs_flat.shape[0])
        numnan_flat = torch.isnan(obs_flat).sum(-1)
        if not set(numnan_flat.unique().tolist()).issubset({0, ndim}):
            raise NotImplementedError

        is_valid = (numnan_flat == 0)
        if is_valid.any():
            is_valid = is_valid.nonzero().unbind(1)
            lp_flat[is_valid] = self.kalman_filter.kf_step.log_prob(
                obs_flat[is_valid],
                means_flat[is_valid],
                covs_flat[is_valid]
            )
        return lp_flat.view(obs.shape[0:2])

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

            stds = torch.diagonal(self.covs, dim1=-1, dim2=-2).sqrt()
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
        raise NotImplementedError
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
