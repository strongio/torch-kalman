from collections import OrderedDict
from typing import Dict
from warnings import warn

import torch

from torch_kalman.covariance import Covariance
from torch_kalman.design_matrix import DesignMatOverTime


class DesignForBatch:
    ok_kwargs = {'input', 'progress'}

    def __init__(self,
                 design: 'Design',
                 num_groups: int,
                 num_timesteps: int,
                 **kwargs):

        self.device = design.device

        # process indices:
        self.process_idx = design.process_idx
        self.process_start_idx = {process_id: idx.start for process_id, idx in self.process_idx.items()}

        # initial mean/cov:
        self.initial_mean = torch.zeros(num_groups, design.state_size, device=self.device)
        init_cov = Covariance.from_log_cholesky(log_diag=design.init_cholesky_log_diag,
                                                off_diag=design.init_cholesky_off_diag,
                                                device=self.device)
        self.initial_covariance = init_cov.expand(num_groups, -1, -1)

        # create processes for batch:
        used_kwargs = set()
        assert isinstance(design.processes, OrderedDict)  # below assumes key ordering
        self.processes = OrderedDict()
        for process_name, process in design.processes.items():
            # kwargs for this process:
            process_kwargs = {k: kwargs.get(k, None) for k in process.expected_batch_kwargs}
            for kwarg in process.expected_batch_kwargs:
                used_kwargs.add(kwarg)

            # assign process:
            self.processes[process_name] = process.for_batch(num_groups=num_groups,
                                                             num_timesteps=num_timesteps,
                                                             **process_kwargs)

            # assign initial mean:
            pslice = self.process_idx[process_name]
            self.initial_mean[:, pslice] = process.initial_state_means_for_batch(design.init_state_mean_params[pslice],
                                                                                 num_groups=num_groups,
                                                                                 **process_kwargs)

        unused_kwargs = (set(kwargs.keys()) - used_kwargs) - self.ok_kwargs
        if unused_kwargs:
            warn(f"Unexpected kwargs:\n{unused_kwargs}.")

        # measures:
        self.measures = design.measures
        self.measure_idx = {measure_id: i for i, measure_id in enumerate(self.measures)}

        # size:
        self.num_groups = num_groups
        self.num_timesteps = num_timesteps
        self.state_size = design.state_size
        self.measure_size = design.measure_size

        # transition-matrix:
        self._transitions = None
        self.F = DesignMatOverTime.from_indices_and_vals(self.transitions,
                                                         size=(self.num_groups, self.state_size, self.state_size),
                                                         device=self.device)

        # measurement-matrix:
        self._state_measurements = None
        self.H = DesignMatOverTime.from_indices_and_vals(self.state_measurements,
                                                         size=(self.num_groups, self.measure_size, self.state_size),
                                                         device=self.device)

        # process covariance matrix:
        self.process_cov_params = {'log_diag': design.process_cholesky_log_diag,
                                   'off_diag': design.process_cholesky_off_diag}
        partial_proc_cov = Covariance.from_log_cholesky(**self.process_cov_params, device=self.device)
        self.Q = DesignMatOverTime.from_smaller_matrix(smaller_mat=partial_proc_cov,
                                                       small_mat_dimnames=list(design.all_dynamic_state_elements()),
                                                       large_mat_dimnames=list(design.all_state_elements()),
                                                       size=(self.num_groups, self.state_size, self.state_size),
                                                       device=self.device)

        # measure-covariance matrix:
        self.measure_cov_params = {'log_diag': design.measure_cholesky_log_diag,
                                   'off_diag': design.measure_cholesky_off_diag}
        R = Covariance.from_log_cholesky(**self.measure_cov_params, device=self.device)
        self.R = DesignMatOverTime(base=R.view(1, *R.shape).expand(self.num_groups, -1, -1))

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
