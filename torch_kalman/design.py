from collections import OrderedDict
from typing import Iterable, Generator

import torch
from torch import Tensor
from torch.nn import Parameter

import numpy as np

from torch_kalman.covariance import Covariance
from torch_kalman.measure import Measure
from torch_kalman.process import Process


class Design:
    def __init__(self, processes: Iterable[Process], measures: Iterable[Measure]):

        # processes:
        self.processes = OrderedDict()
        for process in processes:
            if process.id in self.processes.keys():
                raise ValueError(f"Duplicate process-ids: {process.id}.")
            else:
                self.processes[process.id] = process

        # measures:
        self.measures = OrderedDict()
        for measure in measures:
            if measure.id in self.measures.keys():
                raise ValueError(f"Duplicate measure-ids: {measure.id}.")
            else:
                self.measures[measure.id] = measure

        #
        n = len(self.measures)
        self.measure_cholesky_log_diag = Parameter(data=torch.randn(n))
        self.measure_cholesky_off_diag = Parameter(data=torch.randn(int(n * (n - 1) / 2)))

    def measure_covariance(self):
        return Covariance.from_log_cholesky(log_diag=self.measure_cholesky_log_diag,
                                            off_diag=self.measure_cholesky_off_diag)

    def parameters(self) -> Generator[Parameter, None, None]:
        for process in self.processes.values():
            for param in process.parameters():
                yield param

        yield self.measure_cholesky_log_diag
        yield self.measure_cholesky_off_diag

    def for_batch(self, batch_size: int):
        # TODO: could this be cached? the problem is that we'll end up caching the locked version...
        return DesignForBatch(design=self, batch_size=batch_size)


class DesignForBatch:
    def __init__(self, design: Design, batch_size: int):
        self.batch_size = batch_size
        self.locked = False

        # create processes for batch:
        self._processes = OrderedDict()
        for process_name, process in design.processes.items():
            self.processes[process_name] = process.for_batch(batch_size=batch_size)

        # create measures fo batch:
        self._measures = OrderedDict()
        for measure_name, measure in design.measures.items():
            self.measures[measure_name] = measure.for_batch(batch_size=batch_size)

        # design-mats:
        self.make_measure_covariance = design.measure_covariance
        self._R = None
        self._F = None
        self._H = None
        self._Q = None

    def lock(self):
        self.locked = True

    @property
    def processes(self):
        return self._processes

    @property
    def measures(self):
        return self._measures

    @property
    def R(self) -> Tensor:
        if not self.locked:
            raise Exception("Cannot create/access design-matrices for batch until design is locked.")
        if self._R is None:
            R = self.make_measure_covariance()
            self._R = R.expand(self.batch_size, -1, -1)
        return self._R

    @property
    def F(self) -> Tensor:
        if not self.locked:
            raise Exception("Cannot create/access design-matrices for batch until design is locked.")
        if self._F is None:
            state_size = sum(len(process.state_elements) for process in self.processes.values())
            self._F = torch.zeros((self.batch_size, state_size, state_size))
            start = 0
            for process in self.processes.values():
                end = start + len(process.state_elements)
                self._F[np.ix_(range(self.batch_size), range(start, end), range(start, end))] = process.F
                start = end
        return self._F

    @property
    def Q(self) -> Tensor:
        if not self.locked:
            raise Exception("Cannot create/access design-matrices for batch until design is locked.")
        if self._Q is None:
            state_size = sum(len(process.state_elements) for process in self.processes.values())
            self._Q = torch.zeros((self.batch_size, state_size, state_size))
            start = 0
            for process in self.processes.values():
                end = start + len(process.state_elements)
                self._Q[np.ix_(range(self.batch_size), range(start, end), range(start, end))] = process.Q
                start = end
        return self._Q

    @property
    def H(self) -> Tensor:
        if not self.locked:
            raise Exception("Cannot create/access design-matrices for batch until design is locked.")
        if self._H is None:
            process_lens = [len(process.state_elements) for process in self.processes.values()]
            state_size = sum(process_lens)
            measure_size = len(self.design.measures)
            self._H = torch.zeros((self.batch_size, measure_size, state_size))

            process_start_idx = dict(zip(self.processes.keys(), [0] + process_lens[:-1]))

            for r, (measure_name, measure) in enumerate(self.measures.items()):
                for process_name, measure_vals in measure.processes.items():
                    process = self.processes[process_name]
                    c = process_start_idx[process_name] + process.state_element_idx[process.measurable_state]
                    self._H[:, r, c] = measure_vals

        return self._H
