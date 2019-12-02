from collections import OrderedDict
from copy import copy
from typing import Tuple, Sequence, Dict

import torch

from torch.nn import Parameter, ModuleDict, ParameterDict

from torch_kalman.batch import Batchable
from torch_kalman.covariance import CovarianceFromLogCholesky, PartialCovarianceFromLogCholesky

from torch_kalman.process import Process
from lazy_object_proxy.utils import cached_property

from torch_kalman.process.utils.design_matrix import (
    DynamicMatrix, TransitionMatrix, MeasureMatrix, VarianceMultiplierMatrix
)
from torch_kalman.utils import NiceRepr


class Design(NiceRepr, Batchable):
    _repr_attrs = ('process_list', 'measures')

    def __init__(self, processes: Sequence[Process], measures: Sequence[str], **kwargs):
        """
        :param processes: Processes
        :param measures: Measure-names
        :param kwargs: Not used by this base-class, passed from `KalmanFilter.__init__()`
        """
        self.measures = tuple(measures)

        self.processes = OrderedDict()
        for process in processes:
            if process.id in self.processes.keys():
                raise ValueError(f"Duplicate process-ids: {process.id}.")
            self.processes[process.id] = process

        self._validate()

        # params:
        self._initial_mean = None
        self.init_mean_params = Parameter(torch.randn(len(self.state_elements)))
        self.init_covariance = PartialCovarianceFromLogCholesky(
            full_dim_names=self.state_elements,
            partial_dim_names=self.unfixed_state_elements
        )
        self.process_covariance = PartialCovarianceFromLogCholesky(
            full_dim_names=self.state_elements,
            partial_dim_names=self.dynamic_state_elements
        )
        self.measure_covariance = CovarianceFromLogCholesky(rank=len(self.measures))

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
        out = copy(self)
        out.processes = OrderedDict()
        out.batch_info = (num_groups, num_timesteps)
        out._initial_mean = torch.zeros(num_groups, len(self.state_elements))

        for process_name, process in self.processes.items():
            # get kwargs for this process using sklearn-style disambiguation:
            proc_kwargs = {}
            for k in process.for_batch_kwargs():
                specific_key = "{}__{}".format(process.id, k)
                if specific_key in kwargs:
                    proc_kwargs[k] = kwargs[specific_key]
                elif k in kwargs:
                    proc_kwargs[k] = kwargs[k]

            # wrap calls w/process-name for easier tracebacks:
            try:
                out.processes[process_name] = process.for_batch(
                    num_groups=num_groups,
                    num_timesteps=num_timesteps,
                    **proc_kwargs
                )
                out._initial_mean[:, self.process_slices[process_name]] = process.initial_state_means_for_batch(
                    parameters=self.init_mean_params[self.process_slices[process_name]],
                    num_groups=num_groups,
                    **proc_kwargs
                )
            except Exception as e:
                # add process-name to traceback
                raise type(e)(f"Failed to create `{process}.for_batch()` (see traceback above).") from e

            if out.processes[process_name] is None:
                raise RuntimeError(f"{process_name}'s `for_batch` call did not return anything.")

        return out

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

        p['init_state'] = ParameterDict([('mean', self.init_mean_params)])
        p['init_state'].update(self.init_covariance.param_dict().items())

        p['process_cov'] = self.process_covariance.param_dict()

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
        merged = VarianceMultiplierMatrix.merge(
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
        # base class does not do anything to measure-covariance
        return self._base_R

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
