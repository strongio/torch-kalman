from collections import OrderedDict
from typing import Generator, Tuple, Optional, Dict, Sequence

import torch

from torch import Tensor
from torch.nn import Parameter

from torch_kalman.covariance import Covariance
from torch_kalman.design.for_batch import DesignForBatch
from torch_kalman.process import Process


class Design:
    def __init__(self,
                 processes: Sequence[Process],
                 measures: Sequence[str],
                 device: Optional[torch.device] = None):

        assert processes
        assert measures

        if device is None:
            device = torch.device('cpu')
        self.device = device

        # measures:
        self.measures = tuple(measures)
        assert len(self.measures) == len(set(self.measures)), "Duplicate measures."
        self.measure_size = len(self.measures)

        # add processes:
        self.processes: Dict[str, Process] = OrderedDict()
        for process in processes:

            if process.id in self.processes.keys():
                raise ValueError(f"Duplicate process-ids: {process.id}.")
            else:
                # add process:
                self.processes[process.id] = process

        # now that they're all added, loop through again to get details
        used_measures = set()
        start_counter = 0
        self.process_idx = {}
        self.expected_batch_kwargs = set()
        for process_name, process in self.processes.items():
            end_counter = start_counter + len(process.state_elements)
            self.process_idx[process_name] = slice(start_counter, end_counter)

            # link to design:
            process.link_to_design(self)

            # check measures:
            for measure_id in process.measures:
                if measure_id not in self.measures:
                    raise ValueError(f"Measure '{measure_id}' found in process '{process.id}' but not in `measures`.")
                used_measures.add(measure_id)

            # add expected kwargs
            for kwarg in process.expected_batch_kwargs:
                self.expected_batch_kwargs.add(kwarg)

            #
            start_counter = end_counter
        # noinspection PyUnboundLocalVariable
        self.state_size = end_counter

        # any measures unused?
        unused_measures = set(self.measures).difference(used_measures)
        if unused_measures:
            raise ValueError(f"The following `measures` are not in any of the `processes`:\n{unused_measures}"
                             f"\nUse `Process.add_measure(value=None)` if the measurement-values will be decided per-batch "
                             f"during prediction.")

        #
        self.all_state_elements_idx = {pse: i for i, pse in enumerate(self._all_state_elements())}
        self.measures_idx = {measure: i for i, measure in enumerate(self.measures)}

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
        num_dyn_states = len(list(self._all_dynamic_state_elements()))
        ds_upper_tri = int(num_dyn_states * (num_dyn_states - 1) / 2)
        self.process_cholesky_log_diag = Parameter(torch.zeros(num_dyn_states, device=self.device))
        self.process_cholesky_off_diag = Parameter(torch.zeros(ds_upper_tri, device=self.device))

    def measure_scaling(self) -> Tensor:
        return Covariance.from_log_cholesky(self.measure_cholesky_log_diag,
                                            self.measure_cholesky_off_diag,
                                            device=self.device)

    def _all_state_elements(self) -> Generator[Tuple[str, str], None, None]:
        for process_name, process in self.processes.items():
            for state_element in process.state_elements:
                yield process_name, state_element

    def _all_dynamic_state_elements(self) -> Generator[Tuple[str, str], None, None]:
        for process_name, process in self.processes.items():
            for state_element in process.dynamic_state_elements:
                yield process_name, state_element

    def requires_grad_(self, requires_grad: bool):
        for param in self.parameters():
            param.requires_grad_(requires_grad=requires_grad)

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
