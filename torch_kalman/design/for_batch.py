from collections import OrderedDict
from typing import Sequence, Optional, Tuple, List, Dict
import torch
from torch import Tensor

from torch_kalman.covariance import Covariance

if False:
    from torch_kalman.design import Design  # for type-hinting w/o circular ref

from torch_kalman.process.for_batch import SeqOfTensors, ProcessForBatch


# noinspection PyPep8Naming
class DesignForBatch:
    def __init__(self,
                 design: 'Design',
                 num_groups: int,
                 num_timesteps: int,
                 process_kwargs: Optional[Dict[str, Dict]] = None):
        process_kwargs = process_kwargs or {}

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
        assert set(process_kwargs.keys()).issubset(design.processes.keys())
        assert isinstance(design.processes, OrderedDict)  # below assumes key ordering
        self.processes: Dict[str, ProcessForBatch] = OrderedDict()
        for process_name, process in design.processes.items():
            this_proc_kwargs = process_kwargs.get(process_name, {})

            # assign process:
            try:
                self.processes[process_name] = process.for_batch(num_groups=num_groups,
                                                                 num_timesteps=num_timesteps,
                                                                 **this_proc_kwargs)
            except TypeError as e:
                # if missing kwargs, useful to know which process in the traceback
                raise TypeError("`{pn}.for_batch` raised the following error:\n{e}".format(pn=process_name, e=e))

            # assign initial mean:
            pslice = self.process_idx[process_name]
            self.initial_mean[:, pslice] = process.initial_state_means_for_batch(design.init_state_mean_params[pslice],
                                                                                 num_groups=num_groups,
                                                                                 **this_proc_kwargs)

        # measures:
        self.measure_idx = {measure_id: i for i, measure_id in enumerate(design.measures)}

        # size:
        self.num_groups = num_groups
        self.num_timesteps = num_timesteps
        self.state_size = design.state_size
        self.measure_size = design.measure_size

        # transitions:
        self._F_base: Tensor = None
        self._F_dynamic_assignments: List[Tuple[Tuple[int, int], SeqOfTensors]] = None
        self._F_init()

        # measurements:
        self._H_base: Tensor = None
        self._H_dynamic_assignments: List[Tuple[Tuple[int, int], SeqOfTensors]] = None
        self._H_init()

        # process-var:
        self._Q_base: Tensor = None
        self._Q_diag_multi_dynamic_assignments: List[Tuple[Tuple[int, int], SeqOfTensors]] = None
        self._Q_init()

        # measure-var:
        self._R_base: Tensor = None
        # R_dynamic_assignments not implemented yet
        self._R_init()

    # transitions ----
    def _F_init(self) -> None:
        F_base = torch.zeros(size=(self.num_groups, self.state_size, self.state_size),
                             device=self.device)

        dynamic_assignments = []
        for process_id, process in self.processes.items():
            o = self.process_start_idx[process_id]
            for type, transition_mat_assignments in zip(['base', 'dynamic'], process.transition_mat_assignments):
                for (from_element, to_element), values in transition_mat_assignments.items():
                    r = process.state_element_idx[to_element] + o
                    c = process.state_element_idx[from_element] + o
                    if type == 'dynamic':
                        dynamic_assignments.append(((r, c), values))
                    else:
                        F_base[:, r, c] = values
        self._F_base = F_base
        self._F_dynamic_assignments = dynamic_assignments

    def _F_dynamic(self, base: Tensor, t: int, clone: Optional[bool] = None) -> Tensor:
        if clone or self._F_dynamic_assignments:
            mat = base.clone()
        else:
            mat = base
        for (r, c), values in self._F_dynamic_assignments:
            mat[:, r, c] = values[t]
        return mat

    def F(self, t: int) -> Tensor:
        return self._F_dynamic(base=self._F_base, t=t, clone=None)

    # measurements ----
    def _H_init(self) -> None:
        H_base = torch.zeros(size=(self.num_groups, self.measure_size, self.state_size),
                             device=self.device)
        dynamic_assignments = []
        for process_id, process in self.processes.items():
            o = self.process_start_idx[process_id]
            for type, mmat_assignments in zip(['base', 'dynamic'], process.measurement_mat_assignments):
                for (measure, state_element), values in mmat_assignments.items():
                    r = self.measure_idx[measure]
                    c = process.state_element_idx[state_element] + o
                    if type == 'dynamic':
                        dynamic_assignments.append(((r, c), values))
                    else:
                        H_base[:, r, c] = values
        self._H_base = H_base
        self._H_dynamic_assignments = dynamic_assignments

    def _H_dynamic(self, base: Tensor, t: int, clone: Optional[bool] = None) -> Tensor:
        if clone or self._H_dynamic_assignments:
            mat = base.clone()
        else:
            mat = base
        for (r, c), values in self._H_dynamic_assignments:
            mat[:, r, c] = values[t]
        return mat

    def H(self, t: int) -> Tensor:
        return self._H_dynamic(base=self._H_base, t=t, clone=None)

    # process covariance ----
    def _Q_init(self) -> None:
        partial_proc_cov = Covariance.from_log_cholesky(self.design.process_cholesky_log_diag,
                                                        self.design.process_cholesky_off_diag,
                                                        device=self.device)

        partial_mat_dimnames = list(self.design.all_dynamic_state_elements())
        full_mat_dimnames = list(self.design.all_state_elements())

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
        for process_name, process in self.processes.items():
            measure_idx = [self.measure_idx[m] for m in process.measures]
            log_scaling = measure_log_stds[measure_idx].mean()
            process_slice = self.process_idx[process_name]
            diag_flat[process_slice] = log_scaling.exp()

        diag_multi = torch.diagflat(diag_flat).expand(self.num_groups, -1, -1)
        Q = diag_multi.matmul(Q).matmul(diag_multi)

        # adjustments from processes:
        diag_multi = torch.eye(self.state_size, device=self.device).expand(self.num_groups, -1, -1).clone()
        dynamic_assignments = []
        for process_id, process in self.processes.items():
            o = self.process_start_idx[process_id]
            for type, var_diag_multis in zip(['base', 'dynamic'], process.variance_diag_multi_assignments):
                for state_element, values in var_diag_multis.items():
                    i = process.state_element_idx[state_element] + o
                    if type == 'dynamic':
                        dynamic_assignments.append(((i, i), values))
                    else:
                        diag_multi[:, i, i] = values

        self._Q_base = diag_multi.matmul(Q).matmul(diag_multi)
        self._Q_diag_multi_dynamic_assignments = dynamic_assignments

    def _Q_dynamic(self, base: Tensor, t: int, clone: Optional[bool] = None) -> Tensor:
        if clone or self._Q_diag_multi_dynamic_assignments:
            mat = base.clone()
        else:
            mat = base

        if self._Q_diag_multi_dynamic_assignments:
            diag_multi = torch.eye(self.state_size, device=self.device).expand(self.num_groups, -1, -1).clone()
            for (r, c), values in self._Q_diag_multi_dynamic_assignments:
                diag_multi[:, r, c] = values[t]
            return diag_multi.matmul(mat).matmul(diag_multi)
        else:
            return mat

    def Q(self, t: int) -> Tensor:
        return self._Q_dynamic(base=self._Q_base, t=t, clone=None)

    # measurement covariance ---
    def _R_init(self) -> None:
        R = self.design.measure_scaling()
        self._R_base = R.expand(self.num_groups, -1, -1).clone()

    def _R_dynamic(self, base: Tensor, t: int, clone: Optional[bool] = None):
        if clone:
            mat = base.clone()
        else:
            mat = base
        return mat

    def R(self, t: int) -> Tensor:
        return self._R_dynamic(base=self._R_base, t=t, clone=None)
