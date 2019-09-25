from collections import OrderedDict

import torch

from torch_kalman.process import Process
from torch_kalman.process.utils.design_matrix import (
    DynamicMatrix, TransitionMatrix, MeasureMatrix, VarianceMultiplierMatrix
)
from torch_kalman.utils import cached_property


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

    def _build_processes(self, **kwargs) -> OrderedDict[str, Process]:
        processes = OrderedDict()
        for process_name, process in self.design.processes.items():
            try:
                processes[process_name] = process.for_batch(
                    num_groups=self.design.num_groups,
                    num_timesteps=self.design.num_timesteps,
                    **kwargs
                )
            except (TypeError, ValueError) as e:
                # add process-name to traceback
                raise RuntimeError(f"Failed to create `{process}.for_batch` (see traceback above).") from e

            if processes[process_name] is None:
                raise RuntimeError(f"{process_name}'s `for_batch` call did not return anything.")
        return processes

    def _build_initial_mean(self, **kwargs) -> torch.Tensor:
        init_mean = torch.zeros(self.num_groups, len(self.design.state_elements))
        for process_name, process in self.processes.items():
            init_mean[:, self.design.process_slices[process_name]] = process.initial_state_means_for_batch(
                parameters=self.design.init_state_mean_params[self.design.process_slices[process_name]],
                num_groups=self.num_groups,
                **self._kwargs
            )
        return init_mean

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

    # Initial Cov:
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
        measure_idx = {measure: i for i, measure in enumerate(self.design.measures)}
        measure_log_stds = self.design.measure_covariance.create().diag().sqrt().log()
        diag_flat = torch.ones(len(self.design.state_elements))
        for process_name, process in self.processes.items():
            measure_idx = [measure_idx[m] for m in process.measures]
            diag_flat[self.design.process_slices[process_name]] = measure_log_stds[measure_idx].mean().exp()
        diag_multi = torch.diagflat(diag_flat)
        cov_rescaled = diag_multi.matmul(cov).matmul(diag_multi)
        return cov_rescaled
