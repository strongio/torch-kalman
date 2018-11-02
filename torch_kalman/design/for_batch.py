from collections import OrderedDict
from typing import Dict, Tuple

import torch
from torch import Tensor

from torch_kalman.covariance import Covariance

import numpy as np


class DesignForBatch:
    def __init__(self,
                 design: 'Design',
                 batch_size: int,
                 **kwargs):

        self.batch_size = batch_size
        self.device = design.device

        # create processes for batch:
        self.processes = OrderedDict()
        for process_name, process in design.processes.items():
            process_kwargs = {k: kwargs.get(k, None) for k in process.expected_batch_kwargs}
            self.processes[process_name] = process.for_batch(batch_size=batch_size, **process_kwargs)

        # measures:
        self.measures = design.measures

        # measure-covariance parameters:
        self.measure_cov_params = {'log_diag': design.measure_cholesky_log_diag,
                                   'off_diag': design.measure_cholesky_off_diag}

        # size:
        self.state_size = design.state_size
        self.measure_size = design.measure_size

        # cache the things that can't change:
        self._state_mat_idx = None
        self._Q = None
        self._R = None
        self._F = None
        self._processes_with_batch_transitions = None
        self.cache_design(design)

    @property
    def processes_with_batch_transitions(self):
        if self._processes_with_batch_transitions is None:
            self._processes_with_batch_transitions = [process_id for process_id, process in self.processes.items()
                                                      if process.has_batch_transitions()]
        return self._processes_with_batch_transitions

    def R(self) -> Tensor:
        if self._R is not None:
            return self._R
        R = Covariance.from_log_cholesky(**self.measure_cov_params, device=self.device)
        return R.expand(self.batch_size, -1, -1)


    def F(self) -> Tensor:
        state_mat_idx = self.state_mat_idx()
        if self._F is not None:
            F = self._F.clone()
            for process_id in self.processes_with_batch_transitions:
                F[state_mat_idx[process_id]] = self.processes[process_id].F()
        else:
            F = torch.zeros((self.batch_size, self.state_size, self.state_size), device=self.device)
            for process_id, process in self.processes.items():
                F[state_mat_idx[process_id]] = process.F()
        return F

    def Q(self) -> Tensor:
        if self._Q is not None:
            return self._Q
        Q = torch.zeros((self.batch_size, self.state_size, self.state_size), device=self.device)
        for process_id, idx in self.state_mat_idx().items():
            Q[idx] = self.processes[process_id].Q()
        return Q

    def H(self) -> Tensor:
        H = torch.zeros((self.batch_size, self.measure_size, self.state_size), device=self.device)

        process_lens = [len(process.state_elements) for process in self.processes.values()]
        process_start_idx = dict(zip(self.processes.keys(), np.cumsum([0] + process_lens[:-1])))
        measure_idx = {measure_id: i for i, measure_id in enumerate(self.measures)}

        for process_id, process in self.processes.items():
            for (measure_id, state_element), measure_vals in process.state_elements_to_measures().items():
                if measure_vals is None:
                    raise ValueError(f"The measurement value for measure '{measure_id}' of process '{process_id}' is "
                                     f"None, which means that this needs to be set on a per-batch basis using the "
                                     f"`add_measure` method.")
                r = measure_idx[measure_id]
                c = process_start_idx[process_id] + process.state_element_idx[state_element]
                H[:, r, c] = measure_vals
        return H

    def state_mat_idx(self) -> Dict[str, Tuple]:
        if self._state_mat_idx is not None:
            return self._state_mat_idx
        out = {}
        start = 0
        for process in self.processes.values():
            end = start + len(process.state_elements)
            out[process.id] = np.ix_(range(self.batch_size), range(start, end), range(start, end))
            start = end
        return out

    def cache_design(self, design: 'Design') -> None:
        # TODO: OK with device?
        key = self.batch_size, str(self.device)

        if key not in design.state_mat_idx_cache.keys():
            design.state_mat_idx_cache[key] = self.state_mat_idx()
        self._state_mat_idx = design.state_mat_idx_cache[key]

        if key not in design.Q_cache.keys():
            design.Q_cache[key] = self.Q()
        self._Q = design.Q_cache[key]

        if key not in design.R_cache.keys():
            design.R_cache[key] = self.R()
        self._R = design.R_cache[key]

        if key not in design.F_cache.keys():
            state_mat_idx = self.state_mat_idx()
            F_base = torch.zeros((self.batch_size, self.state_size, self.state_size), device=self.device)
            for process_id, process in self.processes.items():
                if not process.has_batch_transitions():
                    F_base[state_mat_idx[process_id]] = process.F()
            design.F_cache[key] = F_base
        self._F = design.F_cache[key]
