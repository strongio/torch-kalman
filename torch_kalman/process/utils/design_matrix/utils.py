from typing import Sequence, Union, Tuple, List, Callable, Dict

import torch
from torch_kalman.internals.exceptions import InputValidationError

DesignMatAssignment = Union[float, torch.Tensor, Callable]
SeqOfTensors = Union[Tuple[torch.Tensor], List[torch.Tensor]]
DesignMatAdjustment = Union[torch.Tensor, SeqOfTensors]


def adjustments_from_nn(nn: torch.nn.Module,
                        num_groups: int,
                        num_timesteps: int,
                        nn_kwargs: dict,
                        output_names: Sequence[str],
                        time_split_kwargs: Sequence[str] = ()
                        ) -> Dict[str, 'DesignMatAdjustment']:
    """
    An important challenge with using nn-modules to predict outputs is that we pass inputs for the entire group X time
    batch, but we need to split these inputs by time before passing them to the nn's `forward()` method, otherwise the
    backward pass will be very slow due to how PyTorch handles masking gradients. This helper uses heuristics to infer
    which inputs to split.
    """

    if time_split_kwargs:
        if isinstance(time_split_kwargs, str):
            raise ValueError(f"time_split_kwargs `{time_split_kwargs}` needs to be wrapped in a list.")
        nn_outputs = {el: [] for el in output_names}
        for t in range(num_timesteps):
            t_kwargs = nn_kwargs.copy()
            for k in time_split_kwargs:
                t_kwargs[k] = nn_kwargs[k][:, t]
            t_output = nn(**t_kwargs)
            if len(t_output.shape) > 2:
                raise InputValidationError(
                    f"`{nn}` expected to output 2D tensor, instead got {t_output.shape} with given input `{t_kwargs}`"
                )
            if len(t_output.shape) == 1:
                t_output = t_output.unsqueeze(-1)
            for i, el in enumerate(output_names):
                if t_output.shape[1] == 1:
                    nn_outputs[el].append(t_output[:, 0])
                else:
                    nn_outputs[el].append(t_output[:, i])
    else:
        try:
            nn_output = nn(**nn_kwargs)
            if nn_output.shape[0] != num_groups:
                raise InputValidationError(
                    f"Expected {nn} to output a tensor with leading dim length of num_groups ({num_groups}). "
                    f"Input:\n{nn_kwargs}"
                )
            if len(nn_output.shape) == 1:
                nn_output = nn_output.unsqueeze(-1)
            if len(nn_output.shape) > 2:
                raise InputValidationError(
                    f"Expected {nn} to output a 2D tensor. XXX\n"
                    f"Input:\n{nn_kwargs}"
                )
            if nn_output.shape[1] == 1:
                nn_outputs = {el: nn_output[:, 0] for el in output_names}
            elif nn_output.shape[1] == len(output_names):
                nn_outputs = {el: nn_output[:, i] for i, el in enumerate(output_names)}
            else:
                raise InputValidationError(
                    f"Expected {nn} to output a tensor with shape[1] of 1 or {len(output_names)}."
                    f"Input:\n{nn_kwargs}"
                )
        except InputValidationError as e:
            if len(nn_kwargs) == 1:
                input = next(iter(nn_kwargs.values()))
                if input.shape[0] == num_groups:
                    time_split_kwargs = list(nn_kwargs.keys())
                    nn_outputs = adjustments_from_nn(
                        nn=nn,
                        num_groups=num_groups,
                        num_timesteps=num_timesteps,
                        nn_kwargs=nn_kwargs,
                        output_names=output_names,
                        time_split_kwargs=time_split_kwargs
                    )
                    nn._time_split_kwargs = time_split_kwargs
                    return nn_outputs
            raise InputValidationError(
                f"Unable to get an acceptable output from {nn}. (TODO: explain)"
            ) from e

    return nn_outputs
