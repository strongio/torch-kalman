from collections import OrderedDict
from typing import Generator, Tuple, Optional, Dict, Sequence

import torch

from torch.nn import Parameter, ModuleDict, ParameterDict

from torch_kalman.design.for_batch import DesignForBatch
from torch_kalman.covariance import CovarianceFromLogCholesky, PartialCovarianceFromLogCholesky

from torch_kalman.process import Process
from torch_kalman.utils import cached_property


class Design:
    def __init__(self, processes: Sequence[Process], measures: Sequence[str]):
        self.measures = tuple(measures)

        self.processes = OrderedDict()
        for process in processes:
            if process.id in self.processes.keys():
                raise ValueError(f"Duplicate process-ids: {process.id}.")
            self.processes[process.id] = process

        self._validate()

    def param_dict(self) -> ModuleDict:
        p = ModuleDict()
        for process_name, process in self.processes.items():
            p[f"process:{process_name}"] = process.param_dict()

        p['measure_cov'] = self.measure_covariance.param_dict

        p['init_state'] = ParameterDict([('mean', self.init_mean_params)])
        p['init_state'].update(self.init_covariance.param_dict.items())

        p['process_cov'] = self.process_covariance.param_dict

        return p

    @cached_property
    def init_mean_params(self):
        state_size = len(self.state_elements)
        return Parameter(torch.randn(state_size))

    @cached_property
    def init_covariance(self):
        return PartialCovarianceFromLogCholesky(
            full_dim_names=self.state_elements,
            partial_dim_names=self.unfixed_state_elements
        )

    @cached_property
    def process_covariance(self):
        return PartialCovarianceFromLogCholesky(
            full_dim_names=self.state_elements,
            partial_dim_names=self.dynamic_state_elements
        )

    @cached_property
    def measure_covariance(self) -> CovarianceFromLogCholesky:
        return CovarianceFromLogCholesky(rank=len(self.measures))

    @cached_property
    def state_elements(self) -> Sequence[Tuple[str, str]]:
        out = []
        for process_name, process in self.processes.items():
            out.extend((process_name, state_element) for state_element in process.state_elements)
        return out

    @cached_property
    def dynamic_state_elements(self) -> Sequence[Tuple[str, str]]:
        out = []
        for process_name, process in self.processes.items():
            out.extend((process_name, state_element) for state_element in process.dynamic_state_elements)
        return out

    @cached_property
    def unfixed_state_elements(self) -> Sequence[Tuple[str, str]]:
        out = []
        for process_name, process in self.processes.items():
            out.extend((process_name, state_element) for state_element in process.state_elements
                       if state_element not in process.fixed_state_elements)
        return out

    @cached_property
    def process_slices(self) -> OrderedDict[str, slice]:
        process_slices = OrderedDict()
        start_counter = 0
        for process_name, process in self.processes.items():
            end_counter = start_counter + len(process.state_elements)
            process_slices[process_name] = slice(start_counter, end_counter)
            start_counter = end_counter
        return process_slices

    def requires_grad_(self, requires_grad: bool):
        for param in self.param_dict().values():
            param.requires_grad_(requires_grad=requires_grad)

    def for_batch(self,
                  num_groups: int,
                  num_timesteps: int,
                  **kwargs) -> 'DesignForBatch':
        return DesignForBatch(
            design=self,
            num_groups=num_groups,
            num_timesteps=num_timesteps,
            **kwargs
        )

    def _validate(self):
        if not self.measures:
            raise ValueError("Empty `measures`")
        if len(self.measures) != len(set(self.measures)):
            raise ValueError("Duplicates in `measures`")
        if not self.processes:
            raise ValueError("Empty `processes`")

        used_measures = set()
        for process_name, process in self.processes.items():
            for measure in process.measures:
                if measure not in self.measures:
                    raise RuntimeError(f"{measure} not in `measures`")
                used_measures.add(process.measures)

        unused_measures = set(self.measures).difference(used_measures)
        if unused_measures:
            raise ValueError(f"The following `measures` are not in any of the `processes`:\n{unused_measures}")

    def __repr__(self):
        return f"{self.__class__.__name__}(processes={list(self.processes.values())}, measures={self.measures})"
