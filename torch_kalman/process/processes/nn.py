from typing import Generator, Tuple, Optional

import torch

from torch import Tensor
from torch.nn import Parameter

from torch_kalman.covariance import Covariance
from torch_kalman.process import Process
from torch_kalman.process.for_batch import ProcessForBatch
from torch_kalman.utils import itervalues_sorted_keys


class NN(Process):
    """
    Uses a torch.nn.module to map an input tensor into a lower-dimensional state representation.
    """

    def __init__(self,
                 id: str,
                 input_dim: int,
                 state_dim: int,
                 nn_module: torch.nn.Module,
                 process_variance: bool = False,
                 add_module_params_to_process: bool = True,
                 model_mat_kwarg_name: Optional[str] = None):

        self.add_module_params_to_process = add_module_params_to_process
        self.input_dim = input_dim
        self.nn_module = nn_module

        #
        pad_n = len(str(state_dim))
        state_elements = [str(i).rjust(pad_n, "0") for i in range(state_dim)]
        transitions = {el: {el: 1.0} for el in state_elements}

        # process covariance:
        self._dynamic_state_elements = []
        if process_variance:
            self._dynamic_state_elements = state_elements

        # super:
        super().__init__(id=id, state_elements=state_elements, transitions=transitions)

        # expected kwargs
        model_mat_kwarg_name = model_mat_kwarg_name or id  # use the id if they didn't specify
        self.expected_batch_kwargs = (model_mat_kwarg_name,)

    @property
    def dynamic_state_elements(self):
        return self._dynamic_state_elements

    # noinspection PyMethodOverriding
    def add_measure(self, measure: str) -> None:
        for state_element in self.state_elements:
            super().add_measure(measure=measure, state_element=state_element, value=None)

    def parameters(self) -> Generator[Parameter, None, None]:
        if self.add_module_params_to_process:
            yield from self.nn_module.parameters()

    def for_batch(self, num_groups: int, num_timesteps: int, **kwargs) -> ProcessForBatch:
        # validate args:
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

        # need to split nn-output by each output element and by time
        # we don't want to call the nn once then slice it, because then autograd will be forced to keep track of large
        # amounts of unnecessary zero-gradients. so instead, call multiple times.
        for_batch = super().for_batch(num_groups, num_timesteps)

        nn_outputs = {el: [] for el in self.state_elements}
        for t in range(num_timesteps):
            nn_output = self.nn_module(nn_input[:, t, :])
            for i, el in enumerate(self.state_elements):
                nn_outputs[el].append(nn_output[:, i])

        for measure in self.measures:
            for el in self.state_elements:
                for_batch.add_measure(measure=measure, state_element=el, values=nn_outputs[el])

        return for_batch
