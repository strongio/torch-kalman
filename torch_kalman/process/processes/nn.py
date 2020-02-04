from typing import Generator, Optional, Callable, Iterable, Sequence

import torch
from torch.nn import Parameter, ParameterDict

from torch_kalman.process import Process

from torch_kalman.internals.utils import zpad, infer_forward_kwargs
from torch_kalman.process.utils.design_matrix.utils import adjustments_from_nn


class NN(Process):
    """
    Uses a torch.nn.module to map an input tensor into a lower-dimensional state representation.
    """
    batch_kwargs_aliases = {'input': 'predictors'}

    def __init__(self,
                 id: str,
                 input_dim: int,
                 state_dim: int,
                 nn: torch.nn.Module,
                 process_variance: bool = False,
                 init_variance: bool = True,
                 add_module_params_to_process: bool = True,
                 inv_link: Optional[Callable] = None,
                 time_split_kwargs: Sequence[str] = ()):
        """

        :param id:
        :param input_dim:
        :param state_dim:
        :param nn:
        :param process_variance:
        :param init_variance:
        :param add_module_params_to_process:
        :param inv_link:
        :param time_split_kwargs:
        """
        self.inv_link = inv_link

        self.add_module_params_to_process = add_module_params_to_process
        self.input_dim = input_dim
        self.nn = nn
        if not hasattr(self.nn, '_forward_kwargs'):
            self.nn._forward_kwargs = infer_forward_kwargs(nn)
        if not hasattr(self.nn, '_time_split_kwargs'):
            assert set(time_split_kwargs).issubset(self.nn._forward_kwargs)
            self.nn._time_split_kwargs = time_split_kwargs

        #
        self._has_process_variance = process_variance
        self._has_init_variance = init_variance

        #
        pad_n = len(str(state_dim))
        super().__init__(id=id, state_elements=[zpad(i, pad_n) for i in range(state_dim)])

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
            yield from self.nn.parameters()

    def param_dict(self) -> ParameterDict:
        p = ParameterDict()
        if self.add_module_params_to_process:
            for nm, param in self.nn.named_parameters():
                nm = 'module_' + nm.replace('.', '_')
                p[nm] = param
        return p

    def for_batch(self,
                  num_groups: int,
                  num_timesteps: int,
                  allow_extra_timesteps: bool = True,
                  **kwargs) -> 'NN':

        for_batch = super().for_batch(num_groups=num_groups, num_timesteps=num_timesteps)

        if 'num_groups' in self.nn._forward_kwargs:
            kwargs['num_groups'] = num_groups
        if 'num_timesteps' in self.nn._forward_kwargs:
            kwargs['num_timesteps'] = num_timesteps

        adjustments = adjustments_from_nn(
            nn=self.nn,
            num_groups=num_groups,
            num_timesteps=num_timesteps,
            nn_kwargs=kwargs,
            output_names=self.state_elements,
            time_split_kwargs=self.nn._time_split_kwargs
        )
        for measure in self.measures:
            for state_element in self.state_elements:
                for_batch._adjust_measure(
                    measure=measure,
                    state_element=state_element,
                    adjustment=adjustments[state_element],
                    check_slow_grad=False
                )

        return for_batch

    def add_measure(self, measure: str) -> 'NN':
        for se in self.state_elements:
            self._set_measure(measure=measure, state_element=se, value=0., ilink=self.inv_link)
        return self

    def batch_kwargs(self, method: Optional[Callable] = None) -> Iterable[str]:
        return [k for k in self.nn._forward_kwargs if k not in {'num_groups', 'num_timesteps'}]
