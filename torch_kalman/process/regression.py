from typing import Tuple, Sequence, Optional, Union

import torch

from torch import nn

from torch_kalman.process.base import Process
from torch_kalman.process.utils import Identity, Bounded, SingleOutput


class _RegressionBase(Process):
    def __init__(self,
                 id: str,
                 predictors: Sequence[str],
                 h_module: torch.nn.Module,
                 measure: Optional[str] = None,
                 process_variance: bool = False,
                 decay: Optional[Union[nn.Module, Tuple[float, float]]] = None):
        if decay is None:
            transitions = {'all_self': torch.ones(len(predictors))}
        else:
            if isinstance(decay, tuple):
                decay = SingleOutput(numel=len(predictors), transform=Bounded(*decay))
            transitions = nn.ModuleDict({'all_self': decay})

        super().__init__(
            id=id,
            measure=measure,
            state_elements=predictors,
            f_tensors=transitions if decay is None else None,
            f_modules=None if decay is None else transitions,
            h_module=h_module,
            h_kwarg='X',
            time_varying_kwargs=['X'],
            no_pcov_state_elements=[] if process_variance else predictors
        )


class LinearModel(_RegressionBase):
    """
    A process which takes a model-matrix of predictors, and each state corresponds to the coefficient on each.
    """

    def __init__(self,
                 id: str,
                 predictors: Sequence[str],
                 measure: Optional[str] = None,
                 process_variance: bool = False,
                 decay: Optional[Tuple[float, float]] = None):
        """
        :param id: Unique identifier for the process
        :param predictors: A sequence of strings with predictor-names.
        :param measure: The name of the measure for this process.
        :param process_variance: TODO
        :param decay: TODO
        """
        super().__init__(
            id=id,
            predictors=predictors,
            measure=measure,
            h_module=Identity(),
            process_variance=process_variance,
            decay=decay
        )


class NN(_RegressionBase):
    """
    A process which takes a model-matrix of predictors and feeds them into a neural-network; the output of this is then
    used in the KalmanFilter's observation matrix. This allows the KalmanFilter to have states corresponding to
    arbitrary combinations of predictors.
    """

    def __init__(self,
                 id: str,
                 nn: torch.nn.Module,
                 measure: Optional[str] = None,
                 process_variance: bool = False,
                 decay: Optional[Tuple[float, float]] = None):
        """

        :param id: Unique identifier for the process
        :param nn: A `nn.Module` that takes inputs from a model-matrix and tranlates them into entries in the
        observation matrix H.
        :param measure: The name of the measure for this process.
        :param process_variance: TODO
        :param decay: TODO
        """
        num_outputs = self._infer_num_outputs(nn)
        super().__init__(
            id=id,
            predictors=[f'nn{i}' for i in range(num_outputs)],
            h_module=nn,
            measure=measure,
            process_variance=process_variance,
            decay=decay
        )

    @staticmethod
    def _infer_num_outputs(nn: torch.nn.Module) -> int:
        num_weights = False
        if hasattr(nn, 'out_features'):
            return nn.out_features
        try:
            reversed_nn = reversed(nn)
        except TypeError as e:
            if 'not reversible' not in str(e):
                raise e
            reversed_nn = []
        for layer in reversed_nn:
            try:
                num_weights = layer.out_features
                break
            except AttributeError:
                pass
        if num_weights is not False:
            return num_weights
        raise TypeError(
            f"Unable to infer num-outputs of {nn} by iterating over it and looking for the final `out_features`."
        )
