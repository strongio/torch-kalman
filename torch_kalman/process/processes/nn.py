from typing import Generator, Optional, Callable

import torch

from torch import Tensor
from torch.nn import Parameter, ParameterDict

from torch_kalman.process import Process


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
                 inv_link: Optional[Callable] = None):

        self.inv_link = inv_link

        self.add_module_params_to_process = add_module_params_to_process
        self.input_dim = input_dim
        self.nn_module = nn_module

        #
        pad_n = len(str(state_dim))
        super().__init__(id=id,
                         state_elements=[str(i).rjust(pad_n, "0") for i in range(state_dim)])

        for se in self.state_elements:
            self._set_transition(from_element=se, to_element=se, value=1.0)

        # process covariance:
        self._dynamic_state_elements = []
        if process_variance:
            self._dynamic_state_elements = self.state_elements

    @property
    def dynamic_state_elements(self):
        return self._dynamic_state_elements

    def parameters(self) -> Generator[Parameter, None, None]:
        if self.add_module_params_to_process:
            yield from self.nn_module.parameters()

    def param_dict(self) -> ParameterDict:
        p = ParameterDict()
        if self.add_module_params_to_process:
            p['module'] = _module_to_param_dict(self.nn_module)
        return p

    # noinspection PyMethodOverriding
    def for_batch(self, num_groups: int, num_timesteps: int, nn_input: Tensor):
        for_batch = super().for_batch(num_groups, num_timesteps)

        if not isinstance(nn_input, Tensor):
            raise ValueError(f"Process {self.id} received 'nn_input' that is not a Tensor.")
        elif nn_input.requires_grad:
            raise ValueError(f"Process {self.id} received 'nn_input' that requires_grad, which is not allowed.")
        elif torch.isnan(nn_input).any():
            raise ValueError(f"Process {self.id} received 'lm_input' that has nans.")

        mm_num_groups, mm_num_ts, mm_num_preds = nn_input.shape
        assert mm_num_groups == num_groups, f"Batch-size is {num_groups}, but nn_input.shape[0] is {mm_num_groups}."
        assert mm_num_ts == num_timesteps, f"Batch num. timesteps is {num_timesteps}, but nn_input.shape[1] is {mm_num_ts}."
        assert mm_num_preds == self.input_dim, f"Expected {self.input_dim} dim, but nn_input.shape[2] = {mm_num_preds}."

        # need to split nn-output by each output element and by time
        # we don't want to call the nn once then slice it, because then autograd will be forced to keep track of large
        # amounts of unnecessary zero-gradients. so instead, call multiple times.
        nn_outputs = {el: [] for el in self.state_elements}
        for t in range(num_timesteps):
            nn_output = self.nn_module(nn_input[:, t, :])
            for i, el in enumerate(self.state_elements):
                nn_outputs[el].append(nn_output[:, i])

        for measure in self.measures:
            for el in self.state_elements:
                for_batch.adjust_measure(measure=measure, state_element=el, adjustment=nn_outputs[el], check_slow_grad=False)

        return for_batch

    def add_measure(self, measure: str) -> 'NN':
        for se in self.state_elements:
            self._set_measure(measure=measure, state_element=se, value=0., inv_link=self.inv_link)
        return self


def _module_to_param_dict(module: torch.nn.Module) -> ParameterDict:
    out = torch.nn.ParameterDict(module._parameters)
    if len(module._modules) == 0:
        return out.update(module.named_parameters())
    return out.update({nm: _module_to_param_dict(sub_module) for nm, sub_module in module._modules.items()})
