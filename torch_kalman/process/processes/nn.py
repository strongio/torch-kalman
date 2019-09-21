from typing import Generator, Optional, Callable

import torch

from torch import Tensor
from torch.nn import Parameter, ParameterDict

from torch_kalman.process import Process
from torch_kalman.process.mixins.has_predictors import HasPredictors


class NN(HasPredictors, Process):

    def __init__(self,
                 id: str,
                 input_dim: int,
                 state_dim: int,
                 nn_module: torch.nn.Module,
                 process_variance: bool = False,
                 init_variance: bool = True,
                 add_module_params_to_process: bool = True,
                 inv_link: Optional[Callable] = None):
        """
        Uses a torch.nn.module to map an input tensor into a lower-dimensional state representation.
        """

        self.inv_link = inv_link

        self.add_module_params_to_process = add_module_params_to_process
        self.input_dim = input_dim
        self.nn_module = nn_module

        #
        self._has_process_variance = process_variance
        self._has_init_variance = init_variance

        #
        pad_n = len(str(state_dim))
        super().__init__(id=id,
                         state_elements=[str(i).rjust(pad_n, "0") for i in range(state_dim)])

        for se in self.state_elements:
            self._set_transition(from_element=se, to_element=se, value=1.0)

    @property
    def dynamic_state_elements(self):
        return self.state_elements if self._has_process_variance else []

    @property
    def fixed_state_elements(self):
        return [] if self._has_init_variance else self.state_elements

    def parameters(self) -> Generator[Parameter, None, None]:
        if self.add_module_params_to_process:
            yield from self.nn_module.parameters()

    def param_dict(self) -> ParameterDict:
        p = ParameterDict()
        if self.add_module_params_to_process:
            for nm, param in self.nn_module.named_parameters():
                nm: str = nm
                nm = 'module_' + nm.replace('.', '_')
                p[nm] = param
        return p

    def for_batch(self,
                  num_groups: int,
                  num_timesteps: int,
                  predictors: Tensor,
                  allow_extra_timesteps: bool = False) -> 'NN':
        for_batch = super().for_batch(num_groups, num_timesteps,
                                      expected_num_predictors=self.input_dim,
                                      allow_extra_timesteps=allow_extra_timesteps)

        if predictors.shape[1] > num_timesteps:
            predictors = predictors[:, 0:num_timesteps, :]

        # need to split nn-output by each output element and by time
        # we don't want to call the nn once then slice it, because then autograd will be forced to keep track of large
        # amounts of unnecessary zero-gradients. so instead, call multiple times.
        nn_outputs = {el: [] for el in self.state_elements}
        for t in range(num_timesteps):
            nn_output = self.nn_module(predictors[:, t, :])
            for i, el in enumerate(self.state_elements):
                nn_outputs[el].append(nn_output[:, i])

        for measure in self.measures:
            for el in self.state_elements:
                for_batch.adjust_measure(measure=measure, state_element=el, adjustment=nn_outputs[el],
                                         check_slow_grad=False)

        return for_batch

    def add_measure(self, measure: str) -> 'NN':
        for se in self.state_elements:
            self._set_measure(measure=measure, state_element=se, value=0., inv_link=self.inv_link)
        return self
