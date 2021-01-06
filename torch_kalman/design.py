from collections import OrderedDict
from copy import copy
from typing import Tuple, Sequence, Dict, Iterable, Union
from warnings import warn

import torch

from torch.nn import Parameter, ModuleDict, ParameterDict

from torch_kalman.internals.batch import Batchable
from torch_kalman.covariance import CovarianceFromLogCholesky, PartialCovarianceFromLogCholesky
from torch_kalman.internals.utils import infer_forward_kwargs

from torch_kalman.process import Process
from lazy_object_proxy.utils import cached_property

from torch_kalman.process.utils.design_matrix import (
    DynamicMatrix,
    TransitionMatrix,
    MeasureMatrix,
    ProcessVarianceMultiplierMatrix,
    MeasureVarianceMultiplierMatrix
)
from torch_kalman.internals.repr import NiceRepr
from torch_kalman.process.utils.design_matrix.utils import adjustments_from_nn
from torch_kalman.utils.nn import NamedEmbedding
from torch_kalman.utils.nn.fourier_season import FourierSeasonNN


class Design(NiceRepr, Batchable):
    """
    A class for specifying the 'design' of a KalmanFilter -- i.e. what measures are modeled by what processes.
    """
    _repr_attrs = ('process_list', 'measures')

    def __init__(self,
                 processes: Sequence[Process],
                 measures: Sequence[str],
                 measure_var_predict: Sequence[torch.nn.Module] = (),
                 process_var_predict: Sequence[torch.nn.Module] = ()
                 ):
        """
        :param processes: Processes
        :param measures: Measure-names
        :param measure_var_predict: See documentation for KalmanFilter.
        :param process_var_predict: See documentation for KalmanFilter.
        """
        if isinstance(measures, str):
            raise ValueError("Expected `measures` to be a sequence of strings, not a string.")
        self.measures = tuple(measures)
        for m in self.measures:
            if not isinstance(m, str):
                raise ValueError(f"`{m}` is an element of measures, but is not a string")

        self.processes = OrderedDict()
        for process in processes:
            if process.id in self.processes.keys():
                raise ValueError(f"Duplicate process-ids: {process.id}.")
            self.processes[process.id] = process

        self._validate()

        # process-variance predictions:
        self._process_var_nn = self._standardize_var_nn(process_var_predict, var_type='process', top_level=True)

        # measure-variance predictions:
        self._measure_var_nn = self._standardize_var_nn(measure_var_predict, var_type='measure', top_level=True)

        # params:

        # initial:
        self._initial_mean = None
        self.init_covariance = PartialCovarianceFromLogCholesky(
            full_dim_names=self.state_elements,
            partial_dim_names=self.unfixed_state_elements
        )

        # process:
        self.process_covariance = PartialCovarianceFromLogCholesky(
            full_dim_names=self.state_elements,
            partial_dim_names=self.dynamic_state_elements
        )

        # measure:
        self.measure_covariance = CovarianceFromLogCholesky(rank=len(self.measures))
        self._measure_var_adjustments = MeasureVarianceMultiplierMatrix(self.measures)

    @cached_property
    def state_elements(self) -> Sequence[Tuple[str, str]]:
        out = []
        for process_name, process in self.processes.items():
            out.extend((process_name, state_element) for state_element in process.state_elements)
        return out

    @cached_property
    def dynamic_state_elements(self) -> Sequence[Tuple[str, str]]:
        out = []
        for process_name, process in self.processes.items():
            out.extend((process_name, state_element) for state_element in process.dynamic_state_elements)
        return out

    @cached_property
    def unfixed_state_elements(self) -> Sequence[Tuple[str, str]]:
        out = []
        for process_name, process in self.processes.items():
            out.extend((process_name, state_element) for state_element in process.state_elements
                       if state_element not in process.fixed_state_elements)
        return out

    @cached_property
    def process_slices(self) -> Dict[str, slice]:
        process_slices = OrderedDict()
        start_counter = 0
        for process_name, process in self.processes.items():
            end_counter = start_counter + len(process.state_elements)
            process_slices[process_name] = slice(start_counter, end_counter)
            start_counter = end_counter
        return process_slices

    def _validate(self):
        if not self.measures:
            raise ValueError("Empty `measures`")
        if len(self.measures) != len(set(self.measures)):
            raise ValueError("Duplicates in `measures`")
        if not self.processes:
            raise ValueError("Empty `processes`")

        used_measures = set()
        for process_name, process in self.processes.items():
            for measure in process.measures:
                if measure not in self.measures:
                    raise RuntimeError(f"{measure} not in `measures`")
                used_measures.add(measure)

        unused_measures = set(self.measures).difference(used_measures)
        if unused_measures:
            raise ValueError(f"The following `measures` are not in any of the `processes`:\n{unused_measures}")

    # For Batch -------:
    def for_batch(self, num_groups: int, num_timesteps: int, **kwargs) -> 'Design':
        for_batch = copy(self)
        for_batch.processes = OrderedDict()
        for_batch.batch_info = (num_groups, num_timesteps)
        for_batch._initial_mean = torch.zeros(num_groups, len(self.state_elements))

        batch_dim_kwargs = {'num_groups': num_groups, 'num_timesteps': num_timesteps}

        unused_kwargs = set(kwargs.keys())

        # processes:
        for process_name, process in self.processes.items():
            # wrap calls w/process-name for easier tracebacks:
            try:
                # for batch:
                proc_kwargs, used = self._parse_kwargs(
                    batch_kwargs=process.batch_kwargs(),
                    prefix=process.id,
                    all_kwargs=kwargs,
                    aliases=getattr(process, 'batch_kwargs_aliases', {})
                )
                for k in used:
                    unused_kwargs.discard(k)
                proc_for_batch = process.for_batch(**batch_dim_kwargs, **proc_kwargs)
                assert proc_for_batch
                for_batch.processes[process_name] = proc_for_batch

                # init mean:
                init_mean_kwargs, used = self._parse_kwargs(
                    batch_kwargs=process.init_mean_kwargs(),
                    prefix=process.id + "_init_mean",
                    all_kwargs=kwargs,
                    aliases=getattr(process, 'init_mean_kwargs_aliases', {})
                )
                for k in used:
                    unused_kwargs.discard(k)
                for_batch._initial_mean[:, self.process_slices[process_name]] = process.initial_state_means_for_batch(
                    num_groups=num_groups,
                    **init_mean_kwargs
                )
            except Exception as e:
                # add process-name to traceback
                raise type(e)(f"Failed to create `{process}.for_batch()` (see traceback above).") from e

        # var adjustments:
        for_batch._measure_var_adjustments = self._measure_var_adjustments.for_batch(**batch_dim_kwargs)
        for var_type, nn_list in {'measure': self._measure_var_nn, 'process': self._process_var_nn}.items():
            for i, nn in enumerate(nn_list):
                nn_kwargs, used = self._parse_kwargs(
                    prefix=f'{var_type}_var_nn{i}',
                    batch_kwargs=nn._forward_kwargs,
                    all_kwargs={**kwargs, **batch_dim_kwargs},
                    aliases=getattr(nn, '_forward_kwargs_aliases', {})
                )

                # a cheat that makes the `seasonal` alias more convenient:
                if 'datetimes' in nn._forward_kwargs and 'datetimes' not in nn_kwargs and hasattr(nn, '_dt_helper'):
                    if 'start_datetimes' in kwargs:
                        nn_kwargs['datetimes'] = nn._dt_helper.make_grid(kwargs['start_datetimes'], num_timesteps)

                for k in used:
                    unused_kwargs.discard(k)

                try:
                    adjustments = adjustments_from_nn(
                        nn=nn,
                        **batch_dim_kwargs,
                        nn_kwargs=nn_kwargs,
                        output_names=self.measures if var_type == 'measure' else self.dynamic_state_elements,
                        time_split_kwargs=getattr(nn, '_time_split_kwargs', ())
                    )
                except TypeError as e:
                    if "forward() missing 1 required" in str(e):
                        raise TypeError(
                            f"`{var_type}_var_nn.forward()` didn't get an expected argument; see traceback."
                        ) from e
                    raise e

                for el, adj in adjustments.items():
                    for_batch._adjust_variance(el, adjustment=adj, check_slow_grad=False)

        if unused_kwargs:
            warn("Unexpected keyword arguments: {}".format(unused_kwargs))

        return for_batch

    @property
    def initial_mean(self):
        if self.is_for_batch:
            return self._initial_mean
        else:
            raise RuntimeError(
                f"Tried to access `{type(self).__name__}.initial_mean`, but only possible for output of `for_batch()`."
            )

    # Parameters -------:
    def param_dict(self) -> ModuleDict:
        p = ModuleDict()
        for process_name, process in self.processes.items():
            p[f"process:{process_name}"] = process.param_dict()

        p['measure_cov'] = self.measure_covariance.param_dict()
        p['measure_var_nn'] = self._measure_var_nn

        p['init_state'] = self.init_covariance.param_dict()

        p['process_cov'] = self.process_covariance.param_dict()
        p['process_var_nn'] = self._process_var_nn

        return p

    # Transition Matrix -------:
    @cached_property
    def F(self) -> DynamicMatrix:
        merged = TransitionMatrix.merge([(nm, process.transition_mat) for nm, process in self.processes.items()])
        assert list(merged.from_elements) == list(self.state_elements) == list(merged.to_elements)
        return merged.compile()

    # Measurement Matrix ------:
    @cached_property
    def H(self) -> DynamicMatrix:
        merged = MeasureMatrix.merge([(nm, process.measure_mat) for nm, process in self.processes.items()])
        assert list(merged.state_elements) == list(self.state_elements)
        # order dim:
        assert set(merged.measures) == set(self.measures)
        merged.measures[:] = self.measures
        return merged.compile()

    # Process-Covariance Matrix ------:
    def Q(self, t: int) -> torch.Tensor:
        # processes can apply multipliers to the variance of their state-elements:
        diag_multi = self._process_variance_multi(t=t)
        return diag_multi.matmul(self._base_Q).matmul(diag_multi)

    @cached_property
    def _process_variance_multi(self) -> DynamicMatrix:
        merged = ProcessVarianceMultiplierMatrix.merge(
            [(nm, process.variance_multi_mat) for nm, process in self.processes.items()]
        )
        assert list(merged.state_elements) == list(self.state_elements)
        return merged.compile()

    @cached_property
    def _base_Q(self):
        Q = self.process_covariance.create(leading_dims=())

        # process covariance is scaled by the variances of the measurement-variances:
        Q_rescaled = self._scale_covariance(Q)

        # expand for batch-size:
        return Q_rescaled.expand(self.num_groups, -1, -1)

    # Measure-Covariance Matrix ------:
    def R(self, t: int):
        diag_multi = self._measure_variance_multi(t=t)
        return diag_multi.matmul(self._base_R).matmul(diag_multi)

    @cached_property
    def _measure_variance_multi(self) -> DynamicMatrix:
        return self._measure_var_adjustments.compile()

    @cached_property
    def _base_R(self):
        return self.measure_covariance.create(leading_dims=(self.num_groups,))

    # Initial Cov ------:
    @cached_property
    def initial_covariance(self) -> torch.Tensor:
        init_cov = self.init_covariance.create(leading_dims=())
        # init covariance is scaled by the variances of the measurement-variances:
        init_cov_rescaled = self._scale_covariance(init_cov)
        # expand for batch-size:
        return init_cov_rescaled.expand(self.num_groups, -1, -1)

    def _scale_covariance(self, cov: torch.Tensor) -> torch.Tensor:
        """
        Rescale variances associated with processes (process-covariance or initial covariance) by the
        measurement-variances. Helpful in practice for training.
        """
        measure_idx_by_measure = {measure: i for i, measure in enumerate(self.measures)}
        measure_log_stds = self.measure_covariance.create().diag().sqrt().log()
        diag_flat = torch.ones(len(self.state_elements))
        for process_name, process in self.processes.items():
            measure_idx = [measure_idx_by_measure[m] for m in process.measures]
            diag_flat[self.process_slices[process_name]] = measure_log_stds[measure_idx].mean().exp()
        diag_multi = torch.diagflat(diag_flat)
        cov_rescaled = diag_multi.matmul(cov).matmul(diag_multi)
        return cov_rescaled

    @property
    def process_list(self):
        return list(self.processes.values())

    # Private -----:
    def _parse_kwargs(self,
                      prefix: str,
                      all_kwargs: dict,
                      batch_kwargs: Iterable[str],
                      aliases: dict) -> Tuple[dict, set]:
        too_generic = {'input', 'x'}

        # use sklearn-style disambiguation:
        used = set()
        out = {}
        for k in batch_kwargs:
            specific_key = "{}__{}".format(prefix, k)
            if specific_key in all_kwargs:
                out[k] = all_kwargs[specific_key]
                used.add(specific_key)
            elif k in all_kwargs:
                if k in too_generic:
                    raise ValueError(
                        f"The argument `{k}` is too generic, so it needs to be passed in a way that specifies which "
                        f"process it should be handed off to (e.g. {specific_key})."
                    )
                out[k] = all_kwargs[k]
                used.add(k)
            else:
                alias = aliases.get(k) or aliases.get(specific_key)
                if alias in all_kwargs:
                    out[k] = all_kwargs[alias]
                    used.add(alias)
        return out, used

    def _standardize_var_nn(self,
                            var_nn: Union[torch.nn.Module, Sequence],
                            var_type: str,
                            top_level: bool = False) -> torch.nn.Module:

        if top_level:
            if isinstance(var_nn, torch.nn.ModuleList):
                return var_nn

            if callable(var_nn):
                # they passed a single NN instead of a list, wrap it:
                var_nn = [var_nn]
            elif len(var_nn) > 0 and isinstance(var_nn[0], str):
                # they passed a single alias instead of a list, wrap it:
                var_nn = [var_nn]

            return torch.nn.ModuleList([self._standardize_var_nn(sub_nn, var_type) for sub_nn in var_nn])
        else:
            if callable(var_nn):
                out_nn = var_nn
            elif isinstance(var_nn, (tuple, list)):
                alias, args_or_kwargs = var_nn
                num_outputs = len(self.measures if var_type == 'measure' else self.dynamic_state_elements)
                if alias == 'per_group' and isinstance(args_or_kwargs, int):
                    args_or_kwargs = (args_or_kwargs,)
                if isinstance(args_or_kwargs, dict):
                    args, kwargs = (), args_or_kwargs.copy()
                else:
                    args, kwargs = args_or_kwargs, {}

                if alias == 'per_group':
                    if 'embedding_dim' not in kwargs:
                        kwargs['embedding_dim'] = num_outputs
                    out_nn = NamedEmbedding(*args, **kwargs)
                    out_nn._forward_kwargs_aliases = {'input': 'group_names'}
                elif alias == 'seasonal':
                    out_nn = FourierSeasonNN(*args, **kwargs, num_outputs=num_outputs)
                    out_nn._time_split_kwargs = ['datetimes']
                else:
                    raise ValueError(f"Known aliases are 'per_group' and 'seasonal'; got '{alias}'")
            else:
                raise TypeError(
                    f"Expected `{var_type}_var_nn` to be a callable/torch.nn.Module, or a tuple with format "
                    f"`('alias',(arg1,arg2,...)`. Instead got `{type(var_nn)}`."
                )
            if not hasattr(out_nn, '_forward_kwargs'):
                out_nn._forward_kwargs = infer_forward_kwargs(out_nn)
            if not hasattr(out_nn, '_forward_kwargs_aliases'):
                out_nn._forward_kwargs_aliases = {}
            return out_nn

    def _adjust_variance(self,
                         *args,
                         adjustment: 'DesignMatAdjustment',
                         check_slow_grad: bool = True,
                         ):
        if len(args) == 1:
            if isinstance(args[0], (list, tuple)):
                args = args[0]
        if len(args) == 1:
            assert args[0] in self.measures
            self._measure_var_adjustments.adjust(value=adjustment, check_slow_grad=check_slow_grad, measure=args[0])
        else:
            process, state_element = args
            self.processes[process]._adjust_variance(
                state_element=state_element, adjustment=adjustment, check_slow_grad=check_slow_grad
            )
