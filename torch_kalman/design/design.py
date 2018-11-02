from collections import OrderedDict
from typing import Iterable, Generator, Tuple, Optional

import torch

from torch import Tensor
from torch.nn import Parameter

import numpy as np

from torch_kalman.covariance import Covariance
from torch_kalman.design.for_batch import DesignForBatch
from torch_kalman.process import Process


class Design:
    def __init__(self,
                 processes: Iterable[Process],
                 measures: Iterable[str],
                 device: Optional[torch.device] = None):

        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = device

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
                # add process:
                self.processes[process.id] = process
                # increase state-size:
                self.state_size += len(process.state_elements)
                # check measures:
                for measure_id, _ in process.state_elements_to_measures.keys():
                    if measure_id not in self.measures:
                        raise ValueError(f"Measure '{measure_id}' found in process '{process.id}' but not in `measures`.")
                    used_measures.add(measure_id)

        # some processes need to know about the design they're in:
        for process_name in self.processes.keys():
            self.processes[process_name].link_to_design(self)

        # any measures unused?
        unused_measures = set(self.measures).difference(used_measures)
        if unused_measures:
            raise ValueError(f"The following `measures` are not in any of the `processes`:\n{unused_measures}"
                             f"\nUse `Process.add_measure(value=None)` if the measurement-values will be decided per-batch "
                             f"during prediction.")

        # measure-covariance:
        self.measure_cholesky_log_diag = Parameter(data=torch.zeros(self.measure_size))
        self.measure_cholesky_off_diag = Parameter(data=torch.zeros(int(self.measure_size * (self.measure_size - 1) / 2)))

        # cache:
        self.Q_cache, self.R_cache, self.F_cache, self.state_mat_idx_cache = None, None, None, None
        self.reset_cache()

    def all_state_elements(self) -> Generator[Tuple[str, str], None, None]:
        for process_name, process in self.processes.items():
            for state_element in process.state_elements:
                yield process_name, state_element

    def requires_grad_(self, requires_grad):
        for param in self.parameters():
            param.requires_grad_(requires_grad=requires_grad)

    @property
    def requires_grad(self):
        return any(param.requires_grad for param in self.parameters())

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
        self.F_cache = {}
        self.state_mat_idx_cache = {}

    def get_block_diag_initial_state(self, batch_size: int, **kwargs) -> Tuple[Tensor, Tensor]:
        means = torch.zeros((batch_size, self.state_size))
        covs = torch.zeros((batch_size, self.state_size, self.state_size))

        start = 0
        for process_id, process in self.processes.items():
            process_kwargs = {k: kwargs.get(k) for k in process.expected_batch_kwargs}
            process_means, process_covs = process.initial_state(batch_size=batch_size, **process_kwargs)
            end = start + process_means.shape[1]
            means[:, start:end] = process_means
            covs[np.ix_(range(batch_size), range(start, end), range(start, end))] = process_covs
            start = end

        return means, covs
