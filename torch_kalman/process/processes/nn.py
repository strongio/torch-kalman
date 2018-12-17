from typing import Generator, Tuple, Optional

import torch

from torch import Tensor
from torch.nn import Parameter

from torch_kalman.covariance import Covariance
from torch_kalman.process import Process
from torch_kalman.process.for_batch import ProcessForBatch
from torch_kalman.utils import itervalues_sorted_keys


class NN(Process):
    def __init__(self,
                 id: str,
                 input_dim: int,
                 state_dim: int,
                 nn_module: torch.nn.Module,
                 allow_process_variance: bool,
                 add_module_params_to_process: bool = True,
                 model_mat_kwarg_name: Optional[str] = None):

        self.add_module_params_to_process = add_module_params_to_process
        self.input_dim = input_dim
        self.nn_module = nn_module

        # transitions:
        state_elements = [str(i) for i in range(state_dim)]
        transitions = {el: {el: 1.0} for el in state_elements}

        # initial state:
        self.initial_state_mean_params = Parameter(torch.randn(state_dim))
        self.initial_state_cov_params = dict(log_diag=Parameter(data=torch.randn(state_dim)),
                                             off_diag=Parameter(data=torch.randn(int(state_dim * (state_dim - 1) / 2))))

        # process covariance:
        if allow_process_variance:
            self.cov_cholesky_log_diag = Parameter(data=torch.zeros(state_dim))
            self.cov_cholesky_off_diag = Parameter(data=torch.zeros(int(state_dim * (state_dim - 1) / 2)))
        else:
            self.cov_cholesky_log_diag, self.cov_cholesky_off_diag = None, None

        # super:
        super().__init__(id=id, state_elements=state_elements, transitions=transitions)

        # expected kwargs
        model_mat_kwarg_name = model_mat_kwarg_name or id  # use the id if they didn't specify
        self.expected_batch_kwargs = (model_mat_kwarg_name,)

    # noinspection PyMethodOverriding
    def add_measure(self, measure: str) -> None:
        for state_element in self.state_elements:
            self.state_elements_to_measures[(measure, state_element)] = None

    def initial_state(self, **kwargs) -> Tuple[Tensor, Tensor]:
        means = self.initial_state_mean_params
        covs = Covariance.from_log_cholesky(**self.initial_state_cov_params, device=self.device)
        return means, covs

    def parameters(self) -> Generator[Parameter, None, None]:
        yield self.initial_state_mean_params
        for param in itervalues_sorted_keys(self.initial_state_cov_params):
            yield param
        if self.cov_cholesky_off_diag is not None:
            yield self.cov_cholesky_off_diag
        if self.cov_cholesky_log_diag is not None:
            yield self.cov_cholesky_log_diag
        if self.add_module_params_to_process:
            yield from self.nn_module.parameters()

    def covariance(self) -> Covariance:
        if self.cov_cholesky_log_diag is not None:
            return Covariance.from_log_cholesky(log_diag=self.cov_cholesky_log_diag,
                                                off_diag=self.cov_cholesky_off_diag,
                                                device=self.device)
        else:
            ns = len(self.state_elements)
            cov = torch.zeros(size=(ns, ns), device=self.device)
            return cov

    def for_batch(self, num_groups: int, num_timesteps: int, **kwargs) -> ProcessForBatch:
        assert self.state_elements_to_measures, f"HLM process '{self.id}' has no measures."

        argname = self.expected_batch_kwargs[0]
        nn_input = kwargs.get(argname, None)
        if nn_input is None:
            raise ValueError(f"Required argument `{argname}` not found.")
        elif torch.isnan(nn_input).any():
            raise ValueError(f"nans not allowed in `{argname}` tensor")

        mm_num_groups, mm_num_ts, mm_dim = nn_input.shape
        assert mm_num_groups == num_groups, f"Batch-size is {num_groups}, but {argname}.shape[0] is {mm_num_groups}."
        assert mm_num_ts == num_timesteps, f"Batch num. timesteps is {num_timesteps}, but {argname}.shape[1] is {mm_num_ts}."
        assert mm_dim == self.input_dim, f"{argname}.shape[2] = {mm_dim}, but expected self.input_dim, {self.input_dim}."

        # Pdb().set_trace()

        for_batch = super().for_batch(num_groups, num_timesteps)

        nn_outputs = {el: [] for el in self.state_elements}
        for t in range(num_timesteps):
            nn_output = self.nn_module(nn_input[:, t, :])
            for i, el in enumerate(self.state_elements):
                nn_outputs[el].append(nn_output[:, i])

        for measure in self.measures():
            for el in self.state_elements:
                for_batch.add_measure(measure=measure, state_element=el, values=nn_outputs[el])

        return for_batch
