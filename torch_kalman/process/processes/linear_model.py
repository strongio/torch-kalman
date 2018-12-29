from typing import Generator, Tuple, Sequence, Optional, Union

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_kalman.covariance import Covariance
from torch_kalman.process import Process
from torch_kalman.process.for_batch import ProcessForBatch
from torch_kalman.utils import itervalues_sorted_keys, split_flat


class LinearModel(Process):
    def __init__(self,
                 id: str,
                 covariates: Sequence[str],
                 process_variance: Union[bool, Sequence[str]] = False,
                 model_mat_kwarg_name: Optional[str] = None):
        # transitions:
        transitions = {covariate: {covariate: 1.0} for covariate in covariates}

        # process covariance:
        self._dynamic_state_elements = []
        if process_variance:
            self._dynamic_state_elements = covariates if isinstance(process_variance, bool) else process_variance

        # super:
        super().__init__(id=id, state_elements=covariates, transitions=transitions)

        # expected kwargs
        model_mat_kwarg_name = model_mat_kwarg_name or id  # use the id if they didn't specify
        self.expected_batch_kwargs = (model_mat_kwarg_name,)

    @property
    def dynamic_state_elements(self) -> Sequence[str]:
        return self._dynamic_state_elements

    def parameters(self) -> Generator[torch.nn.Parameter, None, None]:
        yield from ()  # no parameters

    # noinspection PyMethodOverriding
    def add_measure(self, measure: str) -> None:
        for state_element in self.state_elements:
            super().add_measure(measure=measure, state_element=state_element, value=None)

    def for_batch(self, num_groups: int, num_timesteps: int, **kwargs) -> ProcessForBatch:
        argname = self.expected_batch_kwargs[0]
        re_model_mat = kwargs.get(argname, None)
        if re_model_mat is None:
            raise ValueError(f"Required argument `{argname}` not found.")
        elif torch.isnan(re_model_mat).any():
            raise ValueError(f"nans not allowed in `{argname}` tensor")

        num_states = len(self.state_elements)
        mm_num_groups, mm_num_ts, mm_num_covs = re_model_mat.shape
        assert mm_num_groups == num_groups, f"Batch-size is {num_groups}, but {argname}.shape[0] is {mm_num_groups}."
        assert mm_num_ts == num_timesteps, f"Batch num. timesteps is {num_timesteps}, but {argname}.shape[1] is {mm_num_ts}."
        assert mm_num_covs == num_states, f"Expected {num_states} covariates, but {argname}.shape[2] = {mm_num_covs}."

        for_batch = super().for_batch(num_groups, num_timesteps)

        for i, covariate in enumerate(self.state_elements):
            for measure in self.measures:
                values = split_flat(re_model_mat[:, :, i], dim=1)
                for_batch.add_measure(measure=measure, state_element=covariate, values=values)

        return for_batch
