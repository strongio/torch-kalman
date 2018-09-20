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
        self.state_size = 0
        self.processes = OrderedDict()
        for process in processes:
            if process.id in self.processes.keys():
                raise ValueError(f"Duplicate process-ids: {process.id}.")
            else:
                self.processes[process.id] = process
                self.state_size += len(process.state_elements)

        # measures:
        self.measures = OrderedDict()
        for measure in measures:
            if measure.id in self.measures.keys():
                raise ValueError(f"Duplicate measure-ids: {measure.id}.")
            else:
                self.measures[measure.id] = measure
        self.measure_size = len(self.measures)

        # measure-covariance:
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

    def for_batch(self, batch_size: int) -> 'DesignForBatch':
        # TODO: could this be cached? the problem is that we'll end up caching the locked version...
        # TODO: seasons are such a common thing to use, should handle it here
        return DesignForBatch(design=self, batch_size=batch_size)


class DesignForBatch:
    def __init__(self, design: Design, batch_size: int):
        self.batch_size = batch_size

        # create processes for batch:
        self.processes = OrderedDict()
        for process_name, process in design.processes.items():
            self.processes[process_name] = process.for_batch(batch_size=batch_size)

        # create measures fo batch:
        self.measures = OrderedDict()
        for measure_name, measure in design.measures.items():
            self.measures[measure_name] = measure.for_batch(batch_size=batch_size)

        # measure-covariance parameters:
        self.measure_cov_params = {'log_diag': design.measure_cholesky_log_diag,
                                   'off_diag': design.measure_cholesky_off_diag}

        # size:
        self.state_size = design.state_size
        self.measure_size = design.measure_size

    def R(self) -> Tensor:
        R = Covariance.from_log_cholesky(**self.measure_cov_params)
        return R.expand(self.batch_size, -1, -1)

    def F(self) -> Tensor:
        F = torch.zeros((self.batch_size, self.state_size, self.state_size))
        start = 0
        for process in self.processes.values():
            end = start + len(process.state_elements)
            F[np.ix_(range(self.batch_size), range(start, end), range(start, end))] = process.F()
            start = end
        return F

    def Q(self) -> Tensor:
        Q = torch.zeros((self.batch_size, self.state_size, self.state_size))
        start = 0
        for process in self.processes.values():
            end = start + len(process.state_elements)
            Q[np.ix_(range(self.batch_size), range(start, end), range(start, end))] = process.Q()
            start = end
        return Q

    def H(self) -> Tensor:
        process_lens = [len(process.state_elements) for process in self.processes.values()]
        H = torch.zeros((self.batch_size, self.measure_size, self.state_size))

        process_start_idx = dict(zip(self.processes.keys(), np.cumsum([0] + process_lens[:-1])))

        for r, (measure_name, measure) in enumerate(self.measures.items()):
            this_measure_processes = measure.processes()
            for process_name, measure_vals in this_measure_processes.items():
                if measure_vals is None:
                    raise ValueError(f"The measurement value for measure '{measure_name}' of process '{process_name}' is "
                                     f"None, which means that this needs to be set on a per-batch basis using the "
                                     f"`add_process` method.")
                try:
                    process = self.processes[process_name]
                except KeyError:
                    raise KeyError(f"The measure '{measure_name}' includes the process '{process_name}', but this process "
                                   f"wasn't passed in the `processes` argument at design init.")
                c = process_start_idx[process_name] + process.state_element_idx[process.measurable_state]
                H[:, r, c] = measure_vals
        return H
