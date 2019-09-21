import inspect

from collections import OrderedDict
from typing import Optional, Tuple, List, Dict, Union, Any
import torch
from torch import Tensor

from torch_kalman.process.for_batch import SeqOfTensors, ProcessForBatch


class DesignForBatch:
    def __init__(self,
                 design: 'Design',
                 num_groups: int,
                 num_timesteps: int,
                 **kwargs):

        self.design = design

        self._processes = None
        self._kwargs = kwargs

        # process indices:
        self.process_idx = design.process_idx
        self.process_start_idx = {process_id: idx.start for process_id, idx in self.process_idx.items()}

        # size:
        self.num_groups = num_groups
        self.num_timesteps = num_timesteps
        self.state_size = design.state_size
        self.measure_size = design.measure_size

        # measures:
        self.measure_idx = {measure_id: i for i, measure_id in enumerate(design.measures)}

        # initial mean/cov:
        self.initial_mean = self._build_init_mean()
        self.initial_covariance = self._build_init_covariance()

        # transitions:
        self._F_base: Tensor = None
        self._F_dynamic_assignments: List[Tuple[Tuple[int, int], SeqOfTensors]] = None

        # measurements:
        self._H_base: Tensor = None
        self._H_dynamic_assignments: List[Tuple[Tuple[int, int], SeqOfTensors]] = None

        # process-var:
        self._Q_base: Tensor = None
        self._Q_diag_multi_dynamic_assignments: List[Tuple[Tuple[int, int], SeqOfTensors]] = None

        # measure-var:
        self._R_base: Tensor = None
        # R_dynamic_assignments not implemented yet

    @property
    def processes(self) -> OrderedDict:
        if self._processes is None:
            processes = OrderedDict()
            for process_name, process in self.design.processes.items():
                try:
                    processes[process_name] = process.for_batch(num_groups=self.num_groups,
                                                                num_timesteps=self.num_timesteps,
                                                                **self._kwargs)
                except (TypeError, ValueError) as e:
                    # add process-name to traceback
                    raise RuntimeError(f"Failed to create `{process}.for_batch` (see traceback above).") from e

                if processes[process_name] is None:
                    raise RuntimeError(f"{process_name}'s `for_batch` call returned None.")
            self._processes = processes
        return self._processes

    # transitions ----
    def _F_init(self) -> None:
        F_base = torch.zeros(size=(self.num_groups, self.state_size, self.state_size))

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
        if self._F_base is None:
            self._F_init()
        return self._F_dynamic(base=self._F_base, t=t, clone=None)

    # measurements ----
    def _H_init(self) -> None:
        H_base = torch.zeros(size=(self.num_groups, self.measure_size, self.state_size))
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
        if self._H_base is None:
            self._H_init()
        return self._H_dynamic(base=self._H_base, t=t, clone=None)

    # process covariance ----
    def _Q_init(self) -> None:
        Q = self.design.process_covariance.create(leading_dims=(self.num_groups,))

        # process variances are scaled by the variances of the measurements they are associated with:
        measure_log_stds = self.design.measure_covariance.create().diag().sqrt().log()
        diag_flat = torch.ones(self.state_size)
        for process_name, process in self.processes.items():
            measure_idx = [self.measure_idx[m] for m in process.measures]
            log_scaling = measure_log_stds[measure_idx].mean()
            process_slice = self.process_idx[process_name]
            diag_flat[process_slice] = log_scaling.exp()

        diag_multi_measure = torch.diagflat(diag_flat).expand(self.num_groups, -1, -1)
        Q = diag_multi_measure.matmul(Q).matmul(diag_multi_measure)

        # adjustments from processes:
        diag_multi_proc = torch.eye(self.state_size).expand(self.num_groups, -1, -1).clone()
        dynamic_assignments = []
        for process_id, process in self.processes.items():
            o = self.process_start_idx[process_id]
            for type, var_diag_multis in zip(['base', 'dynamic'], process.variance_diag_multi_assignments):
                for state_element, values in var_diag_multis.items():
                    i = process.state_element_idx[state_element] + o
                    if type == 'dynamic':
                        dynamic_assignments.append(((i, i), values))
                    else:
                        diag_multi_proc[:, i, i] = values

        self._Q_base = diag_multi_proc.matmul(Q).matmul(diag_multi_proc)
        self._Q_diag_multi_dynamic_assignments = dynamic_assignments

    def _Q_dynamic(self, base: Tensor, t: int, clone: Optional[bool] = None) -> Tensor:
        if clone or self._Q_diag_multi_dynamic_assignments:
            mat = base.clone()
        else:
            mat = base

        if self._Q_diag_multi_dynamic_assignments:
            diag_multi = torch.eye(self.state_size).expand(self.num_groups, -1, -1).clone()
            for (r, c), values in self._Q_diag_multi_dynamic_assignments:
                diag_multi[:, r, c] = values[t]
            return diag_multi.matmul(mat).matmul(diag_multi)
        else:
            return mat

    def Q(self, t: int) -> Tensor:
        if self._Q_base is None:
            self._Q_init()
        return self._Q_dynamic(base=self._Q_base, t=t, clone=None)

    # measurement covariance ---
    def _R_init(self) -> None:
        self._R_base = self.design.measure_covariance.create(leading_dims=(self.num_groups,))

    def _R_dynamic(self, base: Tensor, t: int, clone: Optional[bool] = None):
        if clone:
            mat = base.clone()
        else:
            mat = base
        return mat

    def R(self, t: int) -> Tensor:
        if self._R_base is None:
            self._R_init()
        return self._R_dynamic(base=self._R_base, t=t, clone=None)

    # initial state belief ---
    def _build_init_mean(self) -> Tensor:
        init_mean = torch.zeros(self.num_groups, self.state_size)
        for process_name, process in self.design.processes.items():
            # assign initial mean:
            pslice = self.process_idx[process_name]
            init_mean[:, pslice] = process.initial_state_means_for_batch(self.design.init_state_mean_params[pslice],
                                                                         num_groups=self.num_groups,
                                                                         **self._kwargs)
        return init_mean

    def _build_init_covariance(self) -> Tensor:
        init_cov = self.design.init_covariance.create(leading_dims=(self.num_groups,))

        # init variances are scaled by the variances of the measurements they are associated with:
        measure_log_stds = self.design.measure_covariance.create().diag().sqrt().log()
        diag_flat = torch.ones(self.state_size)
        for process_name, process in self.design.processes.items():
            measure_idx = [self.measure_idx[m] for m in process.measures]
            log_scaling = measure_log_stds[measure_idx].mean()
            process_slice = self.process_idx[process_name]
            diag_flat[process_slice] = log_scaling.exp()

        diag_multi_measure = torch.diagflat(diag_flat).expand(self.num_groups, -1, -1)
        init_cov = diag_multi_measure.matmul(init_cov).matmul(diag_multi_measure)

        return init_cov
