from typing import Generator, Optional, Callable, Iterable

import torch
from torch.nn import Parameter, ParameterDict

from torch_kalman.process import Process
from torch_kalman.process.mixins.has_predictors import HasPredictors
from torch_kalman.internals.utils import zpad


class NN(HasPredictors, Process):
    """
    Uses a torch.nn.module to map an input tensor into a lower-dimensional state representation.
    """

    def __init__(self,
                 id: str,
                 input_dim: int,
                 state_dim: int,
                 nn: torch.nn.Module,
                 process_variance: bool = False,
                 init_variance: bool = True,
                 add_module_params_to_process: bool = True,
                 inv_link: Optional[Callable] = None):
        """

        :param id:
        :param input_dim:
        :param state_dim:
        :param nn:
        :param process_variance:
        :param init_variance:
        :param add_module_params_to_process:
        :param inv_link:
        """
        self.inv_link = inv_link

        self.add_module_params_to_process = add_module_params_to_process
        self.input_dim = input_dim
        self.batch_kwarg = getattr(nn, 'batch_kwarg', 'predictors')
        self.nn = nn

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

        predictor_mat = kwargs.pop(self.batch_kwarg, None)
        if predictor_mat is None:
            raise TypeError("Missing argument: {}".format(self.batch_kwarg))
        if kwargs:
            raise TypeError("Unexpected arguments:{}".format(set(kwargs.keys())))

        for_batch = super().for_batch(
            num_groups=num_groups,
            num_timesteps=num_timesteps
        )

        self._validate_predictor_mat(
            num_groups=num_groups,
            num_timesteps=num_timesteps,
            predictor_mat=predictor_mat,
            expected_num_predictors=self.input_dim,
            allow_extra_timesteps=allow_extra_timesteps
        )

        # TODO: support broadcasting if output shape is (num_groups, 1) or (num_groups, num_times, 1)

        if len(predictor_mat.shape) == 2:
            nn_output = self.nn(predictor_mat)
            nn_outputs = {el: nn_output[i] for i, el in enumerate(self.state_elements)}
        else:
            if predictor_mat.shape[1] > num_timesteps:
                predictor_mat = predictor_mat[:, 0:num_timesteps, :]

            # need to split nn-output by each output element and by time
            # we don't want to call the nn once then slice it, because then autograd will be forced to keep track of
            # large amounts of unnecessary zero-gradients. so instead, call multiple times.
            nn_outputs = {el: [] for el in self.state_elements}
            for t in range(num_timesteps):
                nn_output = self.nn(predictor_mat[:, t, :])
                for i, state_element in enumerate(self.state_elements):
                    nn_outputs[state_element].append(nn_output[:, i])

        for measure in self.measures:
            for state_element in self.state_elements:
                for_batch._adjust_measure(
                    measure=measure,
                    state_element=state_element,
                    adjustment=nn_outputs[state_element],
                    check_slow_grad=False
                )

        return for_batch

    def add_measure(self, measure: str) -> 'NN':
        for se in self.state_elements:
            self._set_measure(measure=measure, state_element=se, value=0., ilink=self.inv_link)
        return self

    def batch_kwargs(self, method: Optional[Callable] = None) -> Iterable[str]:
        return [self.batch_kwarg]
