from collections import OrderedDict
from typing import Iterable, Generator, Tuple, Optional, Dict

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
            device = torch.device('cpu')
        self.device = device

        # measures:
        self.measures = tuple(measures)
        assert len(self.measures) == len(set(self.measures)), "Duplicate measures."
        self.measure_size = len(self.measures)

        # add processes:
        self.processes = OrderedDict()
        for process in processes:
            if process.id in self.processes.keys():
                raise ValueError(f"Duplicate process-ids: {process.id}.")
            else:
                # add process:
                self.processes[process.id] = process

        # now that they're all added, loop through again to get details
        used_measures = set()
        self.state_size = 0
        for process_name in self.processes.keys():
            # link to design:
            self.processes[process_name].link_to_design(self)

            # increase state-size:
            self.state_size += len(self.processes[process_name].state_elements)

            # check measures:
            for measure_id, _ in self.processes[process_name].state_elements_to_measures.keys():
                if measure_id not in self.measures:
                    raise ValueError(f"Measure '{measure_id}' found in process '{process.id}' but not in `measures`.")
                used_measures.add(measure_id)

        # any measures unused?
        unused_measures = set(self.measures).difference(used_measures)
        if unused_measures:
            raise ValueError(f"The following `measures` are not in any of the `processes`:\n{unused_measures}"
                             f"\nUse `Process.add_measure(value=None)` if the measurement-values will be decided per-batch "
                             f"during prediction.")

        # process slices
        self._process_idx = None

        # measure-covariance:
        m_upper_tri = int(self.measure_size * (self.measure_size - 1) / 2)
        self.measure_cholesky_log_diag = Parameter(data=torch.zeros(self.measure_size, device=self.device))
        self.measure_cholesky_off_diag = Parameter(data=torch.zeros(m_upper_tri, device=self.device))

        # initial state:
        s_upper_tri = int(self.state_size * (self.state_size - 1) / 2)
        self.init_state_mean_params = Parameter(torch.randn(self.state_size, device=self.device))
        self.init_cholesky_log_diag = Parameter(torch.zeros(self.state_size, device=self.device))
        self.init_cholesky_off_diag = Parameter(torch.zeros(s_upper_tri, device=self.device))

        #
        num_dyn_states = len(list(self.all_dynamic_state_elements()))
        ds_upper_tri = int(num_dyn_states * (num_dyn_states - 1) / 2)
        self.process_cholesky_log_diag = Parameter(torch.zeros(num_dyn_states, device=self.device))
        self.process_cholesky_off_diag = Parameter(torch.zeros(ds_upper_tri, device=self.device))

    @property
    def process_idx(self) -> Dict[str, slice]:
        if self._process_idx is None:
            process_idx = {}
            start = 0
            for process_id, process in self.processes.items():
                this_end = start + len(process.state_elements)
                process_idx[process_id] = slice(start, this_end)
                start = this_end
            self._process_idx = process_idx
        return self._process_idx

    def all_state_elements(self) -> Generator[Tuple[str, str], None, None]:
        for process_name, process in self.processes.items():
            for state_element in process.state_elements:
                yield process_name, state_element

    def all_dynamic_state_elements(self) -> Generator[Tuple[str, str], None, None]:
        for process_name, process in self.processes.items():
            for state_element in process.dynamic_state_elements:
                yield process_name, state_element

    def requires_grad_(self, requires_grad: bool):
        for param in self.parameters():
            param.requires_grad_(requires_grad=requires_grad)

    def measure_covariance(self) -> Tensor:
        return Covariance.from_log_cholesky(log_diag=self.measure_cholesky_log_diag,
                                            off_diag=self.measure_cholesky_off_diag)

    def parameters(self) -> Generator[Parameter, None, None]:
        for process in self.processes.values():
            for param in process.parameters():
                yield param

        yield self.measure_cholesky_log_diag
        yield self.measure_cholesky_off_diag

        yield self.init_state_mean_params
        yield self.init_cholesky_log_diag
        yield self.init_cholesky_off_diag

        yield self.process_cholesky_log_diag
        yield self.process_cholesky_off_diag

    def for_batch(self, num_groups: int, num_timesteps: int, **kwargs) -> 'DesignForBatch':
        return DesignForBatch(design=self, num_groups=num_groups, num_timesteps=num_timesteps, **kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}(processes={list(self.processes.values())}, measures={self.measures})"
