from collections import OrderedDict
from typing import Tuple

import torch
from torch import Tensor

from torch_kalman.covariance import Covariance

import numpy as np


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
        # measure indices:
        self.measure_idx = {measure_id: i for i, measure_id in enumerate(self.measures)}

        # design-matrices:
        self._H = None
        self._F = None
        self._R = None
        self._Q = None

    @property
    def process_idx(self):
        if self._process_idx is None:
            process_idx = {}
            last_end = 0
            for process_id, process in self.processes.items():
                this_end = last_end + len(process.state_elements)
                process_idx[process_id] = slice(last_end, this_end)
                last_end = this_end
            self._process_idx = process_idx
        return self._process_idx

    @property
    def H(self) -> Tensor:
        if self._H is None:
            H = torch.zeros((self.num_groups, self.num_timesteps, self.measure_size, self.state_size), device=self.device)

            process_start_idx = {process_id: idx.start for process_id, idx in self.process_idx.items()}

            for process_id, process in self.processes.items():
                for (measure_id, state_element), measure_vals in process.state_elements_to_measures().items():
                    if measure_vals is None:
                        raise ValueError(f"The measurement value for measure '{measure_id}' of process '{process_id}' is "
                                         f"None, which means that this needs to be set on a per-batch basis using the "
                                         f"`add_measure` method.")
                    r = self.measure_idx[measure_id]
                    c = process_start_idx[process_id] + process.state_element_idx[state_element]
                    H[:, :, r, c] = measure_vals
            self._H = H
        return self._H

    @property
    def F(self) -> Tensor:
        if self._F is None:
            F = torch.zeros((self.num_groups, self.num_timesteps, self.state_size, self.state_size), device=self.device)

            for process_id, process in self.processes.items():
                F[:, :, self.process_idx[process_id], self.process_idx[process_id]] = process.F()

            self._F = F
        return self._F

    @property
    def Q(self) -> Tensor:
        if self._Q is None:
            Q = torch.zeros((self.num_groups, self.num_timesteps, self.state_size, self.state_size), device=self.device)

            for process_id, process in self.processes.items():
                Q[:, :, self.process_idx[process_id], self.process_idx[process_id]] = process.Q()

            self._Q = Q
        return self._Q

    @property
    def R(self) -> Tensor:
        if self._R is None:
            R = Covariance.from_log_cholesky(**self.measure_cov_params, device=self.device)
            self._R = R.view(1, 1, *R.shape).expand(self.num_groups, self.num_timesteps, -1, -1)
        return self._R

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
