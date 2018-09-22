from collections import OrderedDict
from typing import Iterable, Generator, Dict, Tuple

import torch

from torch import Tensor
from torch.nn import Parameter

import numpy as np

from torch_kalman.covariance import Covariance
from torch_kalman.process import Process


class Design:
    def __init__(self,
                 processes: Iterable[Process],
                 measures: Iterable[str]):

        # measures:
        self.measures = tuple(measures)
        assert len(self.measures) == len(set(self.measures)), "Duplicate measures."
        self.measure_size = len(self.measures)

        # processes:
        used_measures = set()
        self.state_size = 0
        self.processes = OrderedDict()
        for process in processes:
            if process.id in self.processes.keys():
                raise ValueError(f"Duplicate process-ids: {process.id}.")
            else:
                self.processes[process.id] = process
                self.state_size += len(process.state_elements)
                for measure_id, _ in process.measures.keys():
                    if measure_id not in self.measures:
                        raise ValueError(f"Measure '{measure_id}' found in process '{process.id}' but not in `measures`.")
                    used_measures.add(measure_id)
        unused_measures = set(self.measures).difference(used_measures)
        if unused_measures:
            raise ValueError(f"The following `measures` are not in any of the `processes`:\n{unused_measures}"
                             f"\nUse `Process.add_measure(value=None)` if the measurement-values will be decided per-batch "
                             f"during prediction.")

        # measure-covariance:
        self.measure_cholesky_log_diag = Parameter(data=torch.randn(self.measure_size))
        self.measure_cholesky_off_diag = Parameter(data=torch.randn(int(self.measure_size * (self.measure_size - 1) / 2)))

        # cache:
        self.Q_cache, self.R_cache, self.state_mat_idx_cache = None, None, None
        self.reset_cache()

    def measure_covariance(self) -> Tensor:
        return Covariance.from_log_cholesky(log_diag=self.measure_cholesky_log_diag,
                                            off_diag=self.measure_cholesky_off_diag)

    def parameters(self) -> Generator[Parameter, None, None]:
        for process in self.processes.values():
            for param in process.parameters():
                yield param

        yield self.measure_cholesky_log_diag
        yield self.measure_cholesky_off_diag

    def for_batch(self, batch_size: int, time: int, **kwargs) -> 'DesignForBatch':
        if time == 0:
            self.reset_cache()
        return DesignForBatch(design=self, batch_size=batch_size, time=time, **kwargs)

    def reset_cache(self) -> None:
        self.Q_cache = {}
        self.R_cache = {}
        self.state_mat_idx_cache = {}


class DesignForBatch:
    def __init__(self,
                 design: Design,
                 batch_size: int,
                 **kwargs):

        self.batch_size = batch_size

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

        # these can't change from batch to batch, so cache:
        self._state_mat_idx = None
        self._Q = None
        self._R = None
        self.cache_design(design)

    def R(self) -> Tensor:
        if self._R is not None:
            return self._R
        R = Covariance.from_log_cholesky(**self.measure_cov_params)
        return R.expand(self.batch_size, -1, -1)

    def F(self) -> Tensor:
        F = torch.zeros((self.batch_size, self.state_size, self.state_size))
        for process_id, idx in self.state_mat_idx().items():
            F[idx] = self.processes[process_id].F()
        return F

    def Q(self) -> Tensor:
        if self._Q is not None:
            return self._Q
        Q = torch.zeros((self.batch_size, self.state_size, self.state_size))
        for process_id, idx in self.state_mat_idx().items():
            Q[idx] = self.processes[process_id].Q()
        return Q

    def H(self) -> Tensor:
        H = torch.zeros((self.batch_size, self.measure_size, self.state_size))

        process_lens = [len(process.state_elements) for process in self.processes.values()]
        process_start_idx = dict(zip(self.processes.keys(), np.cumsum([0] + process_lens[:-1])))
        measure_idx = {measure_id: i for i, measure_id in enumerate(self.measures)}

        for process_id, process in self.processes.items():
            for (measure_id, state_element), measure_vals in process.measures().items():
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

    def cache_design(self, design: Design) -> None:
        if self.batch_size not in design.state_mat_idx_cache.keys():
            design.state_mat_idx_cache[self.batch_size] = self.state_mat_idx()
        self._state_mat_idx = design.state_mat_idx_cache[self.batch_size]

        if self.batch_size not in design.Q_cache.keys():
            design.Q_cache[self.batch_size] = self.Q()
        self._Q = design.Q_cache[self.batch_size]

        if self.batch_size not in design.R_cache.keys():
            design.R_cache[self.batch_size] = self.R()
        self._R = design.R_cache[self.batch_size]
