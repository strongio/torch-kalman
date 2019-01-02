from collections import OrderedDict
from typing import Sequence
from warnings import warn

import torch
from torch import Tensor

from torch_kalman.covariance import Covariance


# noinspection PyPep8Naming
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

        # design-mats:
        self._design_mat_bases = {}
        self._design_mat_time_mods = {}

        self._init_transition_mats(design)
        self._init_measurement_mats(design)
        self._init_process_cov_mats(design)
        self._init_measure_cov_mats(design)

    def design_mat(self, which: str, t: int) -> Tensor:
        base = self._design_mat_bases[which]
        if not self._design_mat_time_mods[which]:
            return base
        else:
            mat = base.clone()
            for (r, c), values in self._design_mat_time_mods[which]:
                mat[:, r, c] = values[t]
            return mat

    def F(self, t: int) -> Tensor:
        return self.design_mat('F', t)

    def H(self, t: int) -> Tensor:
        return self.design_mat('H', t)

    def Q(self, t: int) -> Tensor:
        return self.design_mat('Q', t)

    def R(self, t: int) -> Tensor:
        return self.design_mat('R', t)

    def _init_transition_mats(self, design: 'Design') -> None:
        F = torch.zeros(size=(self.num_groups, self.state_size, self.state_size),
                        device=self.device)

        dynamic_assignments = []
        for process_id, process in self.processes.items():
            o = self.process_start_idx[process_id]
            for (r, c), values in process.transition_mat_assignments.items():
                if isinstance(values, (tuple, list)):
                    idx = (r + o, c + o)
                    dynamic_assignments.append((idx, values))
                else:
                    F[:, r + o, c + o] = values

        self._design_mat_bases['F'] = F
        self._design_mat_time_mods['F'] = dynamic_assignments

    def _init_measurement_mats(self, design: 'Design') -> None:
        H = torch.zeros(size=(self.num_groups, self.measure_size, self.state_size),
                        device=self.device)
        dynamic_assignments = []
        for process_id, process in self.processes.items():
            o = self.process_start_idx[process_id]
            for (measure_id, c), values in process.measurement_mat_assignments.items():
                r = self.measure_idx[measure_id]
                if isinstance(values, (tuple, list)):
                    idx = (r, c + o)
                    dynamic_assignments.append((idx, values))
                else:
                    H[:, r, c + o] = values

        self._design_mat_bases['H'] = H
        self._design_mat_time_mods['H'] = dynamic_assignments

    def _init_process_cov_mats(self, design: 'Design') -> None:
        partial_proc_cov = Covariance.from_log_cholesky(design.process_cholesky_log_diag,
                                                        design.process_cholesky_off_diag,
                                                        device=self.device)

        partial_mat_dimnames = list(design.all_dynamic_state_elements())
        full_mat_dimnames = list(design.all_state_elements())

        Q = torch.zeros(size=(self.state_size, self.state_size), device=self.device)
        for r in range(len(partial_mat_dimnames)):
            for c in range(len(partial_mat_dimnames)):
                to_r = full_mat_dimnames.index(partial_mat_dimnames[r])
                to_c = full_mat_dimnames.index(partial_mat_dimnames[c])
                Q[to_r, to_c] = partial_proc_cov[r, c]

        self._design_mat_bases['Q'] = Q.expand(self.num_groups, -1, -1)
        self._design_mat_time_mods['Q'] = []

    def _init_measure_cov_mats(self, design: 'Design') -> None:
        R = Covariance.from_log_cholesky(design.measure_cholesky_log_diag,
                                         design.measure_cholesky_off_diag,
                                         device=self.device)
        self._design_mat_bases['R'] = R.expand(self.num_groups, -1, -1)
        self._design_mat_time_mods['R'] = []
