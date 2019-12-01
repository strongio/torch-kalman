import inspect
from collections import OrderedDict
from typing import Dict, Callable

import torch

from torch_kalman.process import Process
from torch_kalman.process.utils.design_matrix import (
    DynamicMatrix, TransitionMatrix, MeasureMatrix, VarianceMultiplierMatrix
)
from lazy_object_proxy.utils import cached_property


class DesignForBatch:
    def __init__(self,
                 design: 'Design',
                 num_groups: int,
                 num_timesteps: int,
                 **kwargs):

        self.design = design
        self.num_groups = num_groups
        self.num_timesteps = num_timesteps
        self.processes = self._build_processes(**kwargs)
        self.initial_mean = self._build_initial_mean(**kwargs)

    # Transition Matrix -------:
    @cached_property
    def F(self) -> DynamicMatrix:
        merged = TransitionMatrix.merge([(nm, process.transition_mat) for nm, process in self.processes.items()])
        assert list(merged.from_elements) == list(self.design.state_elements) == list(merged.to_elements)
        return merged.compile()

    # Measurement Matrix ------:
    @cached_property
    def H(self) -> DynamicMatrix:
        merged = MeasureMatrix.merge([(nm, process.measure_mat) for nm, process in self.processes.items()])
        assert list(merged.state_elements) == list(self.design.state_elements)
        # order dim:
        assert set(merged.measures) == set(self.design.measures)
        merged.measures[:] = self.design.measures
        return merged.compile()

    # Process-Covariance Matrix ------:
    def Q(self, t: int) -> torch.Tensor:
        # processes can apply multipliers to the variance of their state-elements:
        diag_multi = self._process_variance_multi(t=t)
        # TODO: should be a diagonal matrix w/zeros for non-dynamic state-elements
        return diag_multi.matmul(self._base_Q).matmul(diag_multi)

    @cached_property
    def _process_variance_multi(self) -> DynamicMatrix:
        merged = VarianceMultiplierMatrix.merge(
            [(nm, process.variance_multi_mat) for nm, process in self.processes.items()]
        )
        assert list(merged.state_elements) == list(self.design.state_elements)
        return merged.compile()

    @cached_property
    def _base_Q(self):
        Q = self.design.process_covariance.create(leading_dims=())

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
        return self.design.measure_covariance.create(leading_dims=(self.num_groups,))

    # Initial Cov ------:
    @cached_property
    def initial_covariance(self) -> torch.Tensor:
        init_cov = self.design.init_covariance.create(leading_dims=())
        # init covariance is scaled by the variances of the measurement-variances:
        init_cov_rescaled = self._scale_covariance(init_cov)
        # expand for batch-size:
        return init_cov_rescaled.expand(self.num_groups, -1, -1)

    def _scale_covariance(self, cov: torch.Tensor) -> torch.Tensor:
        """
        Rescale variances associated with processes (process-covariance or initial covariance) by the
        measurement-variances. Helpful in practice for training.
        """
        measure_idx_by_measure = {measure: i for i, measure in enumerate(self.design.measures)}
        measure_log_stds = self.design.measure_covariance.create().diag().sqrt().log()
        diag_flat = torch.ones(len(self.design.state_elements))
        for process_name, process in self.processes.items():
            measure_idx = [measure_idx_by_measure[m] for m in process.measures]
            diag_flat[self.design.process_slices[process_name]] = measure_log_stds[measure_idx].mean().exp()
        diag_multi = torch.diagflat(diag_flat)
        cov_rescaled = diag_multi.matmul(cov).matmul(diag_multi)
        return cov_rescaled

    # utils ------:
    def _build_processes(self, **kwargs) -> Dict[str, Process]:
        processes = OrderedDict()
        for process_name, process in self.design.processes.items():
            proc_kwargs = self._get_process_kwargs(process.id, process.for_batch, kwargs)
            try:
                processes[process_name] = process.for_batch(
                    num_groups=self.num_groups,
                    num_timesteps=self.num_timesteps,
                    **proc_kwargs
                )
            except Exception as e:
                # add process-name to traceback
                raise type(e)(f"Failed to create `{process}.for_batch()` (see traceback above).") from e

            if processes[process_name] is None:
                raise RuntimeError(f"{process_name}'s `for_batch` call did not return anything.")
        return processes

    def _build_initial_mean(self, **kwargs) -> torch.Tensor:
        init_mean = torch.zeros(self.num_groups, len(self.design.state_elements))
        for process_name, process in self.processes.items():
            proc_kwargs = self._get_process_kwargs(process.id, process.initial_state_means_for_batch, kwargs)
            init_mean[:, self.design.process_slices[process_name]] = process.initial_state_means_for_batch(
                parameters=self.design.init_mean_params[self.design.process_slices[process_name]],
                num_groups=self.num_groups,
                **proc_kwargs
            )
        return init_mean

    @staticmethod
    def _get_process_kwargs(process_id: str, method: Callable, kwargs: Dict) -> Dict:
        excluded = {'self', 'num_groups', 'num_timesteps'}
        method_keys = []
        for kwarg in inspect.signature(method).parameters:
            if kwarg in excluded:
                continue
            if kwarg == 'kwargs':
                raise ValueError(
                    f"The signature for {method.__qualname__} should not use `**kwargs`, should instead "
                    f"specify keyword-arguments explicitly."
                )
            method_keys.append(kwarg)

        new_kwargs = {key: kwargs[key] for key in ('num_groups', 'num_timesteps') if key in kwargs}
        for key in method_keys:
            specific_key = "{}__{}".format(process_id, key)
            if specific_key in kwargs:
                new_kwargs[key] = kwargs[specific_key]
            elif key in kwargs:
                new_kwargs[key] = kwargs[key]

        return new_kwargs
