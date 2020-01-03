from collections import defaultdict
from typing import Sequence, Dict, Tuple, Union, Optional, Iterable
from warnings import warn

import torch
from lazy_object_proxy.utils import cached_property

from torch import Tensor

from torch_kalman.utils.data import TimeSeriesDataset
from torch_kalman.design import Design
from torch_kalman.state_belief import StateBelief
from torch_kalman.state_belief.base import UnmeasuredError

import numpy as np

from torch_kalman.internals.repr import NiceRepr

Selector = Union[Sequence[int], slice]


class StateBeliefOverTime(NiceRepr):
    """
    The output of the KalmanFilter forward pass, containing a sequence of StateBeliefs which represent one-step-ahead
    predictions.

    Contains methods for evaluating the predictions (log_prob), converting them into dataframes (to_dataframe), and for
    sampling from the underlying distribution (sample_measurements).
    """
    _repr_attrs = ('num_groups', 'num_timesteps')

    def __init__(self, state_beliefs: Sequence['StateBelief'], design: Design):
        """
        :param state_beliefs: A sequence of StateBelief objects, representing one-step-ahead predictions.
        :param design: The design of the kalman-filter that produced these StateBeliefs
        """
        self.state_beliefs = state_beliefs
        self.design = design
        self.family = self.state_beliefs[0].__class__
        self.num_groups = self.state_beliefs[0].num_groups
        self.num_timesteps = len(state_beliefs)

        # the last idx where any updates/predicts occurred:
        self._last_update = None
        self._last_prediction = None
        self.last_predict_idx = -torch.ones(self.num_groups, dtype=torch.int)
        self.last_update_idx = -torch.ones(self.num_groups, dtype=torch.int)
        for t, state_belief in enumerate(state_beliefs):
            self.last_predict_idx[state_belief.last_measured == 1] = t
            self.last_update_idx[state_belief.last_measured == 0] = t

        self._means = None
        self._covs = None
        self._last_measured = None
        self._H = None
        self._R = None

    # Attributes stacked from the contained StateBeliefs ---------:
    @property
    def means(self) -> Tensor:
        if self._means is None:
            self._means_covs()
        return self._means

    @property
    def covs(self) -> Tensor:
        if self._covs is None:
            self._means_covs()
        return self._covs

    @property
    def H(self) -> Tensor:
        if self._H is None:
            self._H = torch.stack([sb.H for sb in self.state_beliefs], 1)
        return self._H

    @property
    def R(self) -> Tensor:
        if self._R is None:
            self._R = torch.stack([sb.R for sb in self.state_beliefs], 1)
        return self._R

    # Information for Prediction ---------:
    @cached_property
    def predictions(self) -> Tensor:
        """
        The predictions on the measurement scale -- i.e., appropriate for comparing to the input that was originally fed
         into the KalmanFilter, e.g. via metrics like MSE.
        """
        return self.H.matmul(self.means.unsqueeze(-1)).squeeze(-1)

    @cached_property
    def prediction_uncertainty(self) -> Tensor:
        """
        Uncertainty on the measurement scale, aka "system uncertainty".
        """
        Ht = self.H.permute(0, 1, 3, 2)
        return self.H.matmul(self.covs).matmul(Ht) + self.R

    def last_update(self) -> StateBelief:
        no_updates = self.last_update_idx < 0
        if no_updates.any():
            raise ValueError(
                f"The following groups have never been updated:\n{no_updates.nonzero().squeeze().tolist()}"
            )
        return self._restore_sb(enumerate(self.last_update_idx.tolist()))

    def last_prediction(self) -> StateBelief:
        no_predicts = self.last_predict_idx < 0
        if no_predicts.any():
            raise ValueError(
                f"The following groups have never been predicted:\n{no_predicts.nonzero().squeeze().tolist()}"
            )
        return self._restore_sb(enumerate(self.last_predict_idx.tolist()))

    def state_belief_for_time(self, times: Sequence[int]) -> StateBelief:
        return self._restore_sb(enumerate(times))

    # Distribution-Methods -----------:
    def log_prob(self, obs: Tensor, **kwargs) -> Tensor:
        """
        Compute the log-probability of data (e.g. data that was originally fed into the KalmanFilter.

        :param obs: A Tensor that could be used in the KalmanFilter.forward pass.
        :param kwargs: Other keyword arguments needed to evaluate the log-prob (e.g. for a censored-kalman-filter, the
          upper and lower bounds).
        :return: A tensor with one element for each group X timestep indicating the log-probability.
        """
        if obs.grad_fn is not None:
            warn("`obs` has a grad_fn, nans may propagate to gradient")

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
                out[group_idx, time_idx] = self._log_prob_with_subsetting(
                    obs=obs,
                    group_idx=group_idx,
                    time_idx=(time_idx,),
                    measure_idx=measure_idx,
                    **kwargs
                ).squeeze(-1)
            else:
                # time has dimension, but group is a slice so it's OK
                out[group_idx, time_idx] = self._log_prob_with_subsetting(
                    obs=obs,
                    group_idx=group_idx,
                    time_idx=time_idx,
                    measure_idx=measure_idx,
                    **kwargs
                )

        return out

    def sample_measurements(self, eps: Optional[Union[Tensor, float]] = None) -> Tensor:
        """
        Generate samples from the underlying torch.Distribution (usually a MultivariateNormal) on the measurement scale.

        :param eps: An optional float that will act as a multiplier on the noise in sampling. For advanced use-cases can
          alternatively be a Tensor that will be used as uncorrelated white noise when generating samples (this can be
          useful e.g. in generating laplace samples from a fitted model).
        :return: A tensor of random samples.
        """
        raise NotImplementedError

    # Exporting to other Formats ---------:
    def to_dataframe(self,
                     dataset: Union[TimeSeriesDataset, dict],
                     type: str = 'predictions',
                     group_colname: str = 'group',
                     time_colname: str = 'time') -> 'DataFrame':
        """
        :param dataset: Either a TimeSeriesDataset, or a dictionary with 'start_times' and 'group_names'.
        :param type: Either 'predictions' or 'components'.
        :param group_colname: Column-name for 'group'
        :param time_colname: Column-name for 'time'
        :return: A pandas DataFrame with group, time, measure, plus:
            - For type='predictions': predicted_mean, predicted_std
            - For type='components': process, state_element, value, std
        """
        from pandas import concat

        if isinstance(dataset, TimeSeriesDataset):
            batch_info = {'start_times': dataset.start_times, 'group_names': dataset.group_names, 'named_tensors': {}}
            for measure_group, tensor in zip(dataset.measures, dataset.tensors):
                for i, measure in enumerate(measure_group):
                    if measure in self.design.measures:
                        batch_info['named_tensors'][measure] = tensor[..., [i]]
            batch_info['times'] = dataset.times()
        elif isinstance(dataset, dict):
            batch_info = dataset
        else:
            raise TypeError(
                "Expected `batch` to be a TimeSeriesDataset, or a dictionary with 'start_times' and 'group_names'."
            )

        def _tensor_to_df(tens, measures):
            times = batch_info.get('times', batch_info['start_times'][:, None] + np.arange(0, self.num_timesteps))
            return TimeSeriesDataset.tensor_to_dataframe(
                tensor=tens,
                times=times,
                group_names=batch_info['group_names'],
                group_colname=group_colname,
                time_colname=time_colname,
                measures=measures
            )

        out = []
        if type == 'predictions':

            stds = torch.diagonal(self.prediction_uncertainty, dim1=-1, dim2=-2).sqrt()
            for i, measure in enumerate(self.design.measures):
                # predicted:
                df_pred = _tensor_to_df(self.predictions[..., [i]], measures=['predicted_mean'])
                df_std = _tensor_to_df(stds[..., [i]], measures=['predicted_std'])
                df = df_pred.merge(df_std, on=[group_colname, time_colname])

                # actual:
                orig_tensor = batch_info.get('named_tensors', {}).get(measure, None)
                if orig_tensor is not None:
                    df_actual = _tensor_to_df(orig_tensor, measures=['actual'])
                    df = df.merge(df_actual, on=[group_colname, time_colname], how='left')

                out.append(df.assign(measure=measure))

        elif type == 'components':
            # components:
            for (measure, process, state_element), (m, std) in self.components().items():
                df = _tensor_to_df(torch.stack([m, std], 2), measures=['value', 'std'])
                df['process'], df['state_element'], df['measure'] = process, state_element, measure
                out.append(df)

            # residuals:
            for measure, orig_tensor in batch_info.get('named_tensors', {}).items():
                if orig_tensor.shape[1] < self.predictions.shape[1]:
                    orig_aligned = self.predictions.data.clone()
                    orig_aligned[:] = float('nan')
                    orig_aligned[:, 0:orig_tensor.shape[1], :] = orig_tensor
                else:
                    orig_aligned = orig_tensor[:, 0:self.predictions.shape[1], :]
                df = _tensor_to_df(self.predictions - orig_aligned, ['value'])
                df['process'], df['state_element'], df['measure'] = 'residuals', 'residuals', measure
                out.append(df)

        else:
            raise ValueError("Expected `type` to be 'predictions' or 'components'.")

        return concat(out, sort=True)

    def components(self) -> Dict[Tuple[str, str, str], Tuple[Tensor, Tensor]]:
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
             multi: float = 1.96,
             max_num_groups: int = 1,
             **kwargs) -> 'DataFrame':
        """
        :param df: The output of `.to_dataframe()`.
        :param group_colname: The name of the group-column.
        :param time_colname: The name of the time-column.
        :param multi: Multiplier to convert std-devs to CIs, default 1.96.
        :param max_num_groups: Max. number of groups to plot; if the number of groups in the dataframe is greater than
        this, a random subset will be taken.
        :param kwargs: Further keyword arguments to pass to `plotnine.theme` (e.g. `figure_size=(x,y)`)
        :return: A plot of the predicted and actual values.
        """

        from plotnine import ggplot, aes, geom_line, geom_ribbon, facet_grid, facet_wrap, theme_bw, theme, ylab

        is_components = ('process' in df.columns and 'state_element' in df.columns)

        if group_colname is None:
            if 'group' in df.columns:
                group_colname = 'group'
            else:
                raise TypeError("Please specify group_colname")
        if time_colname is None:
            if 'time' in df.columns:
                time_colname = 'time'
            else:
                raise TypeError("Please specify time_colname")
        if is_components:
            value_colname = 'value'
            std_colname = 'std'
        else:
            value_colname = 'predicted_mean'
            std_colname = 'predicted_std'

        df = df.copy()
        if df[group_colname].nunique() > max_num_groups:
            subset_groups = df[group_colname].drop_duplicates().sample(max_num_groups).tolist()
            if len(subset_groups) < df[group_colname].nunique():
                print("Subsetting to groups: {}".format(subset_groups))
            df = df.loc[df[group_colname].isin(subset_groups), :]
        num_groups = df[group_colname].nunique()

        df['lower'] = df[value_colname] - multi * df[std_colname]
        df['upper'] = df[value_colname] + multi * df[std_colname]

        aes_kwargs = {'x': time_colname}
        if is_components:
            aes_kwargs['group'] = 'state_element'

        plot = (
                ggplot(df, aes(**aes_kwargs)) +
                geom_line(aes(y=value_colname), color='#4C6FE7', size=1.5, alpha=.75) +
                geom_ribbon(aes(ymin='lower', ymax='upper'), color=None, alpha=.25) +
                ylab("")
        )

        if is_components:
            if num_groups > 1:
                raise ValueError("Cannot plot components for > 1 group.")
            plot = plot + facet_grid(f"process ~ measure", scales='free_y', labeller='label_both')

            if 'figure_size' not in kwargs:
                kwargs['figure_size'] = (12, df['process'].nunique() * 2.5)

        else:
            if 'actual' in df.columns:
                plot = plot + geom_line(aes(y='actual'))
            if num_groups > 1:
                plot = plot + facet_grid(f"{group_colname} ~ measure", scales='free_y', labeller='label_both')
            else:
                plot = plot + facet_wrap("~measure", scales='free_y', labeller='label_both')

            if 'figure_size' not in kwargs:
                kwargs['figure_size'] = (12, 5)

        return plot + theme_bw() + theme(**kwargs)

    # Private utils ---------:
    def _restore_sb(self, indices: Iterable[Tuple[int, int]]) -> StateBelief:
        means, covs, H, R = [], [], [], []
        for g, t in indices:
            means.append(self.means[g, t])
            covs.append(self.covs[g, t])
            H.append(self.H[g, t])
            R.append(self.R[g, t])
        sb = self.family(torch.stack(means), torch.stack(covs))
        try:
            sb.compute_measurement(H=torch.stack(H), R=torch.stack(R))
        except UnmeasuredError:
            pass
        return sb

    @staticmethod
    def _which_valid_key(is_nan: Tensor) -> Tuple[int]:
        num_multi_dims = sum(x > 1 for x in is_nan.shape)
        if num_multi_dims > 1:
            raise ValueError("Expected `tensor` to be 1D (or have only one non-singleton dimension.")
        is_valid = ~is_nan
        return tuple(is_valid.nonzero().squeeze(-1).tolist())

    def _means_covs(self) -> None:
        means, covs = zip(*[(state_belief.means, state_belief.covs) for state_belief in self.state_beliefs])
        self._means = torch.stack(means, 1)
        self._covs = torch.stack(covs, 1)

    def _log_prob_with_subsetting(self,
                                  obs: Tensor,
                                  group_idx: Selector,
                                  time_idx: Selector,
                                  measure_idx: Selector,
                                  **kwargs) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def _check_lp_sub_input(group_idx: Selector, time_idx: Selector):
        if isinstance(group_idx, Sequence) and isinstance(time_idx, Sequence):
            if len(group_idx) > 1 and len(time_idx) > 1:
                warn(
                    "Both `group_idx` and `time_idx` are indices (i.e. neither is an int or a slice). This is rarely "
                    "the expected input."
                )
