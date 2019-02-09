from collections import OrderedDict
from typing import Sequence, Optional
from warnings import warn

import torch
from torch import Tensor

from torch_kalman.covariance import Covariance

# noinspection PyPep8Naming
from torch_kalman.utils import _add


class DesignForBatch:
    ok_kwargs = {'input', 'progress'}

    def __init__(self,
                 design: 'Design',
                 num_groups: int,
                 num_timesteps: int,
                 **kwargs):

        self.design = design
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
            raise RuntimeError(f"Unexpected kwargs:\n{unused_kwargs}.")

        # measures:
        self.measures = design.measures
        self.measure_idx = {measure_id: i for i, measure_id in enumerate(self.measures)}

        # size:
        self.num_groups = num_groups
        self.num_timesteps = num_timesteps
        self.state_size = design.state_size
        self.measure_size = design.measure_size

        # transitions:
        self.F_base = None
        self.F_dynamic_assignments = None
        self.F_init()

        # measurements:
        self.H_base = None
        self.H_dynamic_assignments = None
        self.H_init()

        # process-var:
        self.Q_base = None
        self.Q_diag_multi_dynamic_assignments = None
        self.Q_init()

        # measure-var:
        self.R_base = None
        # R_dynamic_assignments not implemented yet
        self.R_init()

    # transitions ----
    def F_init(self) -> None:
        F = torch.zeros(size=(self.num_groups, self.state_size, self.state_size),
                        device=self.device)
        raise NotImplementedError("where does the link function come in?")

        dynamic_assignments = []
        for process_id, process in self.processes.items():
            o = self.process_start_idx[process_id]
            for type, transition_mat_assignments in zip(['base', 'dynamic'], process.transition_mat_assignments):
                for trans_key, values in transition_mat_assignments.items():
                    r, c = (process.state_element_idx[se] for se in trans_key)
                    if type == 'dynamic':
                        idx = (r + o, c + o)
                        dynamic_assignments.append((idx, values))
                    else:
                        F[:, r + o, c + o] = _add(values)
        self.F_base = F
        self.F_dynamic_assignments = dynamic_assignments

    def F_dynamic(self, base: Tensor, t: int, clone: Optional[bool] = None) -> Tensor:
        if clone or self.F_dynamic_assignments:
            mat = base.clone()
        else:
            mat = base
        for (r, c), list_of_values in self.F_dynamic_assignments:
            mat[:, r, c] = _add(values[t] for values in list_of_values)
        return mat

    def F(self, t: int) -> Tensor:
        return self.F_dynamic(base=self.F_base, t=t, clone=None)

    # measurements ----
    def H_init(self) -> None:
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
        self.H_base = H
        self.H_dynamic_assignments = dynamic_assignments

    def H_dynamic(self, base: Tensor, t: int, clone: Optional[bool] = None) -> Tensor:
        if clone or self.H_dynamic_assignments:
            mat = base.clone()
        else:
            mat = base
        for (r, c), values in self.H_dynamic_assignments:
            mat[:, r, c] = values[t]
        return mat

    def H(self, t: int) -> Tensor:
        return self.H_dynamic(base=self.H_base, t=t, clone=None)

    # process covariance ----
    def Q_init(self) -> None:
        partial_proc_cov = Covariance.from_log_cholesky(self.design.process_cholesky_log_diag,
                                                        self.design.process_cholesky_off_diag,
                                                        device=self.device)

        partial_mat_dimnames = list(self.design._all_dynamic_state_elements())
        full_mat_dimnames = list(self.design._all_state_elements())

        # move from partial cov to full w/block-diag:
        Q = torch.zeros(size=(self.num_groups, self.state_size, self.state_size), device=self.device)
        for r in range(len(partial_mat_dimnames)):
            for c in range(len(partial_mat_dimnames)):
                to_r = full_mat_dimnames.index(partial_mat_dimnames[r])
                to_c = full_mat_dimnames.index(partial_mat_dimnames[c])
                Q[:, to_r, to_c] = partial_proc_cov[r, c]

        # process variances are scaled by the variances of the measurements they are associated with:
        measure_log_stds = self.design.measure_scaling().diag().sqrt().log()
        diag_flat = torch.ones(self.state_size, device=self.device)
        for measure_idx, process_slice in self.design.proc_idx_to_measure_idx:
            log_scaling = measure_log_stds[measure_idx].mean()
            diag_flat[process_slice] = log_scaling.exp()
        diag_multi = torch.diagflat(diag_flat).expand(self.num_groups, -1, -1)
        Q = diag_multi.matmul(Q).matmul(diag_multi)

        # some variances might be inflated/deflated on a per-batch basis:
        diag_multi = torch.eye(self.state_size, device=self.device).expand(self.num_groups, -1, -1).clone()
        dynamic_assignments = []
        for process_id, process in self.processes.items():
            o = self.process_start_idx[process_id]
            for (r, c), values in process.variance_diag_multi_assignments.items():
                if isinstance(values, (tuple, list)):
                    idx = (r + o, c + o)
                    dynamic_assignments.append((idx, values))
                else:
                    diag_multi[:, r + o, c + o] = values

        self.Q_base = diag_multi.matmul(Q).matmul(diag_multi)
        self.Q_diag_multi_dynamic_assignments = dynamic_assignments

    def Q_dynamic(self, base: Tensor, t: int, clone: Optional[bool] = None) -> Tensor:
        if clone or self.Q_diag_multi_dynamic_assignments:
            mat = base.clone()
        else:
            mat = base

        if self.Q_diag_multi_dynamic_assignments:
            diag_multi = torch.eye(self.state_size, device=self.device).expand(self.num_groups, -1, -1).clone()
            for (r, c), values in self.Q_diag_multi_dynamic_assignments:
                diag_multi[:, r, c] = values[t]
            return diag_multi.matmul(mat).matmul(diag_multi)
        else:
            return mat

    def Q(self, t: int) -> Tensor:
        return self.Q_dynamic(base=self.Q_base, t=t, clone=None)

    # measurement covariance ---
    def R_init(self) -> None:
        R = self.design.measure_scaling()
        self.R_base = R.expand(self.num_groups, -1, -1).clone()

    def R_dynamic(self, base: Tensor, t: int, clone: Optional[bool] = None):
        if clone:
            mat = base.clone()
        else:
            mat = base
        return mat

    def R(self, t: int) -> Tensor:
        return self.R_dynamic(base=self.R_base, t=t, clone=None)
