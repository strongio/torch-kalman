from collections import OrderedDict
from typing import Sequence
from warnings import warn

import torch
from torch import Tensor

from torch_kalman.covariance import Covariance


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
        self.F = self._init_transition_mats(num_timesteps)
        self.H = self._init_measurement_mats(num_timesteps)
        self.Q = self._init_process_cov_mats(num_timesteps, design)
        self.R = self._init_measure_cov_mats(num_timesteps, design)

    def _init_transition_mats(self, num_timesteps: int) -> Sequence[Tensor]:
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

        if not dynamic_assignments:
            return [F for _ in range(num_timesteps)]
        else:
            out = []
            for t in range(num_timesteps):
                Ft = F.clone()
                for (r, c), values in dynamic_assignments:
                    Ft[:, r, c] = values[t]
                out.append(Ft)
            return out

    def _init_measurement_mats(self, num_timesteps: int) -> Sequence[Tensor]:
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

        if not dynamic_assignments:
            return [H for _ in range(num_timesteps)]
        else:
            out = []
            for t in range(num_timesteps):
                Ht = H.clone()
                for (r, c), values in dynamic_assignments:
                    Ht[:, r, c] = values[t]
                out.append(Ht)
            return out

    def _init_process_cov_mats(self, num_timesteps: int, design: 'Design') -> Sequence[Tensor]:
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

        Q = Q.expand(self.num_groups, -1, -1)
        return [Q for _ in range(num_timesteps)]

    def _init_measure_cov_mats(self, num_timesteps: int, design: 'Design') -> Sequence[Tensor]:
        R = Covariance.from_log_cholesky(design.measure_cholesky_log_diag,
                                         design.measure_cholesky_off_diag,
                                         device=self.device)
        R = R.expand(self.num_groups, -1, -1)
        return [R for _ in range(num_timesteps)]



