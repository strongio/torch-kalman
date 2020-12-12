from typing import Generator, Optional, Callable, Iterable, Sequence, Union, Tuple

import torch
from torch.nn import Parameter, ParameterDict

from torch_kalman.process import Process

from torch_kalman.internals.utils import zpad, infer_forward_kwargs
from torch_kalman.process.utils.bounded import Bounded
from torch_kalman.process.utils.design_matrix.utils import adjustments_from_nn


class NN(Process):
    """
    Uses a torch.nn.Module to map an input tensor into a lower-dimensional state representation.

    When calling your KalmanFilter, the input to this nn.Module can be passed using the keyword-arguments to its
    `forward()` method. For example, if this NN process's id = 'nn_process', and you pass a single module
    `MyVarPredictModule`, with `forward()` that takes  `input`, you would call your KalmanFilter with
    `nn_process__input=[predictor-tensor]`. Alternatively, the NN process supports using the simpler alias `predictors`.
    """
    batch_kwargs_aliases = {'input': 'predictors'}
    inv_link = None

    def __init__(self,
                 id: str,
                 input_dim: int,
                 state_dim: int,
                 nn: torch.nn.Module,
                 process_variance: bool = False,
                 decay: Union[bool, Tuple[float, float]] = False,
                 time_split_kwargs: Sequence[str] = (),
                 initial_state: Optional[torch.nn.Module] = None):
        """
        :param id: A unique identifier for the process.
        :param input_dim: The number of inputs to the nn.
        :param state_dim: The number of outputs of the nn.
        :param nn: A torch.nn.Module that takes a (num_groups, input_dim) Tensor, and outputs a (num_groups, state_dim)
        Tensor.
        :param process_variance: If False (the default), then the uncertainty about the values of the states does not
        grow at each timestep, so over time these eventually converge to a certain value. If True, then the latent-
        states are allowed to 'drift' over time.
        :param decay: If True, then in forecasts (or for missing data) the state-values will tend to shrink towards
        zero. Usually only used if `process_variance=True`. Default False. Instead of `True` you can specify custom-
        bounds for the decay-rate as a tuple.
        :param time_split_kwargs: When calling the KalmanFilter, you will pass a prediction Tensor for your nn.Module
        that is (num_groups, num_timesteps, input_dim). However, internally, this will be split up into multiple
        tensors, and your nn.Module will take a (num_groups, input_dim) tensor. If your nn.Module's `forward()` method
        takes just a single argument, then we can infer how to split this tensor. But if it takes multiple keyword
        arguments, you need to specify which will be split in this fashion.
        :param initial_state: Optional, a callable (typically a torch.nn.Module). When the KalmanFilter is called,
        keyword-arguments can be passed to initial_state in the format `{this_process}_initial_state__{kwarg}`.
        """

        self.input_dim = input_dim
        self.nn = nn
        if not hasattr(self.nn, '_forward_kwargs'):
            self.nn._forward_kwargs = infer_forward_kwargs(nn)
        if not hasattr(self.nn, '_time_split_kwargs'):
            assert set(time_split_kwargs).issubset(self.nn._forward_kwargs)
            self.nn._time_split_kwargs = time_split_kwargs

        #
        self._has_process_variance = process_variance

        #
        pad_n = len(str(state_dim))
        super().__init__(id=id, state_elements=[zpad(i, pad_n) for i in range(state_dim)], initial_state=initial_state)

        # decay:
        self.decays = {}
        if decay:
            if decay is True:
                self.decays = {se: Bounded(.95, 1.00) for se in self.state_elements}
            else:
                self.decays = {se: Bounded(*decay) for se in self.state_elements}

        for se in self.state_elements:
            decay = self.decays.get(se)
            self._set_transition(from_element=se, to_element=se, value=decay.get_value if decay else 1.0)

    @property
    def dynamic_state_elements(self):
        return self.state_elements if self._has_process_variance else []

    def param_dict(self) -> ParameterDict:
        p = super().param_dict()
        for nm, decay in self.decays.items():
            p[f'decay_{nm}'] = decay.parameter

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

    def batch_kwargs(self) -> Iterable[str]:
        return [k for k in self.nn._forward_kwargs if k not in {'num_groups', 'num_timesteps'}]
