from collections import OrderedDict
from typing import Tuple, Sequence, Dict

import torch
from torch import Tensor

from torch_kalman.covariance import Covariance
from torch_kalman.design_matrix import DesignMatOverTime


class DesignForBatch:
    def __init__(self,
                 design: 'Design',
                 input: Tensor,
                 **kwargs):

        self.device = design.device

        # create processes for batch:
        self.processes = OrderedDict()
        for process_name, process in design.processes.items():
            process_kwargs = {k: kwargs.get(k, None) for k in process.expected_batch_kwargs}
            self.processes[process_name] = process.for_batch(input=input, **process_kwargs)

        # measures:
        self.measures = design.measures

        # measure-covariance parameters:
        self.measure_cov_params = {'log_diag': design.measure_cholesky_log_diag,
                                   'off_diag': design.measure_cholesky_off_diag}

        # size:
        self.state_size = design.state_size
        self.measure_size = design.measure_size
        self.num_groups, self.num_timesteps, *_ = input.shape

        # process indices:
        self._process_idx = None
        self.process_start_idx = {process_id: idx.start for process_id, idx in self.process_idx.items()}
        # measure indices:
        self.measure_idx = {measure_id: i for i, measure_id in enumerate(self.measures)}

        # design-matrices:
        self._transitions = None
        self.F = DesignMatOverTime.from_indices_and_vals(self.transitions,
                                                         size=(self.num_groups, self.state_size, self.state_size),
                                                         device=self.device)

        self._state_measurements = None
        self.H = DesignMatOverTime.from_indices_and_vals(self.state_measurements,
                                                         size=(self.num_groups, self.measure_size, self.state_size),
                                                         device=self.device)

        self._block_diag_covariance = None
        self.Q = DesignMatOverTime.from_indices_and_vals(self.block_diag_covariance,
                                                         size=(self.num_groups, self.state_size, self.state_size),
                                                         device=self.device)

        R = Covariance.from_log_cholesky(**self.measure_cov_params, device=self.device)
        self.R = DesignMatOverTime(base=R.view(1, *R.shape).expand(self.num_groups, -1, -1))

    @property
    def block_diag_covariance(self) -> Sequence[Tuple]:
        if self._block_diag_covariance is None:
            block_diag_covariance = []
            for process_id, process in self.processes.items():
                process_slice = self.process_idx[process_id]
                idx = (process_slice, process_slice)
                block_diag_covariance.append((idx, process.process.covariance()))
            self._block_diag_covariance = block_diag_covariance
        return self._block_diag_covariance

    @property
    def transitions(self) -> Dict:
        if self._transitions is None:
            transitions = {}
            for process_id, process in self.processes.items():
                o = self.process_start_idx[process_id]
                for (r, c), values in process.transitions.items():
                    transitions[r + o, c + o] = values
            self._transitions = transitions
        return self._transitions

    @property
    def state_measurements(self) -> Dict:
        if self._state_measurements is None:
            state_measurements = {}
            for process_id, process in self.processes.items():
                o = self.process_start_idx[process_id]
                for (measure_id, c), values in process.state_measurements.items():
                    state_measurements[self.measure_idx[measure_id], c + o] = values
            self._state_measurements = state_measurements
        return self._state_measurements

    @property
    def process_idx(self) -> Dict[str, slice]:
        if self._process_idx is None:
            process_idx = {}
            last_end = 0
            for process_id, process in self.processes.items():
                this_end = last_end + len(process.state_elements)
                process_idx[process_id] = slice(last_end, this_end)
                last_end = this_end
            self._process_idx = process_idx
        return self._process_idx

    def get_block_diag_initial_state(self, num_groups: int) -> Tuple[Tensor, Tensor]:
        means = torch.zeros((num_groups, self.state_size), device=self.device)
        covs = torch.zeros((num_groups, self.state_size, self.state_size), device=self.device)

        start = 0
        for process_id, process in self.processes.items():
            process_means, process_covs = process.initial_state
            end = start + process_means.shape[1]
            means[:, start:end] = process_means
            covs[:, start:end, start:end] = process_covs
            start = end

        return means, covs
