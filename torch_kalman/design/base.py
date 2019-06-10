from collections import OrderedDict
from typing import Generator, Tuple, Optional, Dict, Sequence

import torch

from torch.nn import Parameter, ModuleDict, ParameterDict

from torch_kalman.design.for_batch import DesignForBatch
from torch_kalman.covariance import CovarianceFromLogCholesky, PartialCovarianceFromLogCholesky

from torch_kalman.process import Process


class Design:
    def __init__(self,
                 processes: Sequence[Process],
                 measures: Sequence[str],
                 **kwargs):

        assert processes
        assert measures

        if kwargs.get('device', 'cpu') != 'cpu' or kwargs.get('dtype', 'float') != 'float':
            raise NotImplementedError("Only device='cpu' and dtype='float' are currently supported.")
        self.device = torch.device('cpu')

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
        self.process_idx: Dict[str, slice] = {}
        for process_name, process in self.processes.items():
            end_counter = start_counter + len(process.state_elements)
            self.process_idx[process_name] = slice(start_counter, end_counter)

            # set device:
            process.set_device(self.device)

            # check measures:
            for measure_id in process.measures:
                if measure_id not in self.measures:
                    raise ValueError(f"Measure '{measure_id}' found in process '{process.id}' but not in `measures`.")
                used_measures.add(measure_id)

            #
            start_counter = end_counter
        # noinspection PyUnboundLocalVariable
        self.state_size = end_counter

        # any measures unused?
        unused_measures = set(self.measures).difference(used_measures)
        if unused_measures:
            raise ValueError(f"The following `measures` are not in any of the `processes`:\n{unused_measures}.")

        #
        self.all_state_elements_idx = {pse: i for i, pse in enumerate(self.all_state_elements())}
        self.measures_idx = {measure: i for i, measure in enumerate(self.measures)}

        # measure-covariance:
        m_upper_tri = int(self.measure_size * (self.measure_size - 1) / 2)
        self.measure_cholesky_log_diag = Parameter(data=.01 * torch.randn(self.measure_size, device=self.device))
        self.measure_cholesky_off_diag = Parameter(data=.01 * torch.randn(m_upper_tri, device=self.device))

        # initial state:
        self.init_state_mean_params = Parameter(torch.randn(self.state_size, device=self.device))
        self._init_covariance = None

        # process cov:
        self._process_covariance = None

        # measure covariance:
        self._measure_covariance = None

    @property
    def init_covariance(self):
        if self._init_covariance is None:
            self._init_covariance = PartialCovarianceFromLogCholesky(full_dim_names=self.all_state_elements(),
                                                                     partial_dim_names=self.all_unfixed_state_elements())
        return self._init_covariance

    @property
    def process_covariance(self):
        if self._process_covariance is None:
            self._process_covariance = PartialCovarianceFromLogCholesky(full_dim_names=self.all_state_elements(),
                                                                        partial_dim_names=self.all_dynamic_state_elements())
        return self._process_covariance

    @property
    def measure_covariance(self):
        if self._measure_covariance is None:
            self._measure_covariance = CovarianceFromLogCholesky(rank=len(self.measures))
        return self._measure_covariance

    def all_state_elements(self) -> Generator[Tuple[str, str], None, None]:
        for process_name, process in self.processes.items():
            for state_element in process.state_elements:
                yield process_name, state_element

    def all_dynamic_state_elements(self) -> Generator[Tuple[str, str], None, None]:
        for process_name, process in self.processes.items():
            for state_element in process.dynamic_state_elements:
                yield process_name, state_element

    def all_unfixed_state_elements(self) -> Generator[Tuple[str, str], None, None]:
        for process_name, process in self.processes.items():
            fixed = set(process.fixed_state_elements)
            for state_element in process.state_elements:
                if state_element not in fixed:
                    yield process_name, state_element

    def requires_grad_(self, requires_grad: bool):
        for param in self.param_dict().values():
            param.requires_grad_(requires_grad=requires_grad)

    def param_dict(self) -> ModuleDict:
        p = ModuleDict()
        for process_name, process in self.processes.items():
            p[f"process:{process_name}"] = process.param_dict()

        p['measure_cov'] = self.measure_covariance.param_dict

        p['init_state'] = ParameterDict([('mean', self.init_state_mean_params)])
        p['init_state'].update(self.init_covariance.param_dict)

        p['process_cov'] = self.process_covariance.param_dict

        return p

    def for_batch(self,
                  num_groups: int,
                  num_timesteps: int,
                  process_kwargs: Optional[Dict[str, Dict]] = None) -> 'DesignForBatch':
        return DesignForBatch(design=self,
                              num_groups=num_groups,
                              num_timesteps=num_timesteps,
                              process_kwargs=process_kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}(processes={list(self.processes.values())}, measures={self.measures})"
