from typing import Tuple, Sequence, Optional, List

import torch

from torch import jit, nn, Tensor

from torch_kalman.rewrite.helpers import Identity, ReturnValues, SimpleTransition


class Process(jit.ScriptModule):
    def __init__(self,
                 id: str,
                 state_elements: Sequence[str],
                 transitions: torch.nn.ModuleDict,
                 h_module: nn.Module):
        super(Process, self).__init__()
        self.id = id

        self.state_elements = state_elements
        self.h_module = h_module
        self.transition_modules = transitions
        self._validate()

        self.se_to_idx = {se: i for i, se in enumerate(self.state_elements)}

        self.measure = ''

        # elements without process covariance, defaults to none
        self.no_pcov_state_elements: List[str] = []

    def get_groupwise_args(self, *args, **kwargs) -> List[Tensor]:
        raise NotImplementedError

    def get_timewise_args(self, *args, **kwargs) -> List[Tensor]:
        raise NotImplementedError

    def get_num_groups_from_inputs(self, inputs: List[Tensor]) -> int:
        # TODO: get rid of this
        raise NotImplementedError

    def set_measure(self, measure: str) -> 'Process':
        self.measure = measure
        return self

    @jit.script_method
    def forward(self, inputs: List[Tensor]) -> Tuple[Tensor, Tensor]:
        H = self.h_forward(inputs)
        F = self.f_forward(inputs)
        return H, F

    def h_forward(self, inputs: List[Tensor]) -> Tensor:
        return self.h_module(inputs)

    def f_forward(self, inputs: List[Tensor]) -> Tensor:
        F = jit.annotate(List[Tensor], [])
        for to_el in self.state_elements:
            for from_el in self.state_elements:
                F += [self._get_transition(inputs, from_el, to_el)]
        return torch.stack(F).view(len(self.state_elements), len(self.state_elements))

    def _get_transition(self, inputs: List[Tensor], from_el: str, to_el: str) -> Tensor:
        num_groups = self.get_num_groups_from_inputs(inputs)
        # jit doesn't support getitem
        for from__to, module in self.transition_modules.items():
            if from__to == f"{from_el}->{to_el}":
                value = module(inputs)
                if len(value.shape) == 1 and num_groups is not None:
                    # TODO: what if output is `(num_groups,)` instead of `(num_groups,1)`
                    value = value.expand(num_groups, -1)
                return value
        if num_groups is None:
            return torch.zeros(1)
        else:
            return torch.zeros(num_groups, 1)

    def _validate(self):
        for se in self.state_elements:
            if '.' in se:
                raise ValueError(f"State-elements cannot contain '->', got '{se}'.")

        for from__to in self.transition_modules.keys():
            from_el, _, to_el = from__to.partition("->")
            if not to_el:
                raise ValueError(f"`transitions` must be '->'-delimited `state_elements`, but {from__to} is not.")
            if from_el not in self.state_elements or to_el not in self.state_elements:
                raise ValueError(f"One or both of the state-elements in `{from__to}` is not in `{self.state_elements}`")


class LocalLevel(Process):
    def get_groupwise_args(self, *args, **kwargs) -> List[Tensor]:
        return []

    def get_timewise_args(self, *args, **kwargs) -> List[Tensor]:
        return []

    def get_num_groups_from_inputs(self, inputs: List[Tensor]) -> Optional[int]:
        return None

    def __init__(self, id: str, decay: Optional[Tuple[float, float]] = None):
        """
        :param id: A unique identifier for this process.
        :param decay: If the process has decay, then the random walk will tend towards zero as we forecast out further
        (note that this means you should center your time-series, or you should include another process that does not
        have this property). Decay can be between 0 and 1, but values < .50 (or even .90) can often be too rapid and
        you will run into trouble with vanishing gradients. When passing a pair of floats, the nn.Module will assign a
        parameter representing the decay as a learned parameter somewhere between these bounds.
        """
        transitions = nn.ModuleDict()
        if decay is None:
            decay = (None, 1.0)
        transitions['position->position'] = SimpleTransition(decay)
        super(LocalLevel, self).__init__(
            id=id,
            state_elements=['position'],
            transitions=transitions,
            h_module=ReturnValues(torch.tensor([1.]))
        )


class LocalTrend(Process):

    def get_groupwise_args(self, *args, **kwargs) -> List[Tensor]:
        return []

    def get_timewise_args(self, *args, **kwargs) -> List[Tensor]:
        return []

    def get_num_groups_from_inputs(self, inputs: List[Tensor]) -> Optional[int]:
        return None

    def __init__(self,
                 id: str,
                 decay_velocity: Optional[Tuple[float, float]] = (.95, 1.00),
                 decay_position: Optional[Tuple[float, float]] = None,
                 velocity_multi: float = 1.0):
        """
        :param id: A unique identifier for this process.
        :param decay_velocity: If set, then the trend will decay to zero as we forecast out further. The default is
        to allow the trend to decay somewhere between .95 (moderate decay) and 1.00 (no decay), with the exact value
         being a learned parameter.
        :param decay_position: See `decay` in `LocalLevel`. Default is no decay.
        :param velocity_multi: A multiplier on the velocity, so that
        `next_position = position + velocity_multi * velocity`. A value of << 1.0 can be helpful since the
        trend has such a large effect on the prediction, so that large values can lead to exploding predictions.
        """

        # define transitions:
        transitions = nn.ModuleDict()
        if decay_position is None:
            decay_position = (None, 1.0)
        transitions['position->position'] = SimpleTransition(decay_position)
        if decay_velocity is None:
            decay_velocity = (None, 1.0)
        transitions['velocity->velocity'] = SimpleTransition(decay_velocity)
        assert velocity_multi <= 1.
        transitions['velocity->position'] = SimpleTransition((None, velocity_multi))

        super(LocalTrend, self).__init__(
            id=id,
            state_elements=['position', 'velocity'],
            transitions=transitions,
            h_module=ReturnValues(torch.tensor([1., 0.]))
        )


class PredictorBase(Process):
    mm_kwarg_name = 'predictors'

    def get_groupwise_args(self, *args, **kwargs) -> List[Tensor]:
        return []

    def get_timewise_args(self, *args, **kwargs) -> List[Tensor]:
        predictor_mat = kwargs.get(f"{self.id}__{self.mm_kwarg_name}", kwargs.get(self.mm_kwarg_name))
        if predictor_mat is None:
            raise RuntimeError(f"Process `{self.id}` expected a keyword argument '{self.mm_kwarg_name}'.")
        return [predictor_mat]

    def get_num_groups_from_inputs(self, inputs: List[Tensor]) -> Optional[int]:
        return inputs[0].shape[0]

    def __init__(self,
                 id: str,
                 predictors: Sequence[str],
                 h_module: torch.nn.Module,
                 process_variance: bool,
                 decay: Optional[Tuple[float, float]]):
        transitions = nn.ModuleDict()
        for pred in predictors:
            transitions[f'{pred}->{pred}'] = SimpleTransition(decay or (None, 1.0))
        super(PredictorBase, self).__init__(
            id=id,
            state_elements=predictors,
            transitions=transitions,
            h_module=h_module
        )


class LinearModel(PredictorBase):
    def __init__(self,
                 id: str,
                 predictors: Sequence[str],
                 process_variance: bool = False,
                 decay: Optional[Tuple[float, float]] = None):
        super(LinearModel, self).__init__(
            id=id,
            predictors=predictors,
            h_module=Identity(),
            process_variance=process_variance,
            decay=decay
        )
        if not process_variance:
            self.no_pcov_state_elements = self.state_elements


class NN(PredictorBase):
    def __init__(self,
                 id: str,
                 nn: torch.nn.Module,
                 process_variance: bool = False,
                 decay: Optional[Tuple[float, float]] = None):
        num_outputs = self._infer_num_outputs(nn)
        super(NN, self).__init__(
            id=id,
            predictors=[f'nn{i}' for i in range(num_outputs)],
            h_module=nn,
            process_variance=process_variance,
            decay=decay
        )
        if not process_variance:
            self.no_pcov_state_elements = self.state_elements

    @staticmethod
    def _infer_num_outputs(nn: torch.nn.Module) -> int:
        num_weights = False
        for layer in reversed(nn):
            try:
                num_weights = layer.out_features
                break
            except AttributeError:
                pass
        if num_weights is not False:
            return num_weights
        raise TypeError(
            "Unable to infer num-outputs of the nn by iterating over it and looking for the final `out_features`."
        )
