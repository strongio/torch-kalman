from typing import Sequence, Optional, Union, Callable

import torch

from torch import Tensor

from torch_kalman.process import Process
from torch_kalman.process.for_batch import ProcessForBatch


class LinearModel(Process):
    def __init__(self,
                 id: str,
                 covariates: Sequence[str],
                 process_variance: Union[bool, Sequence[str]] = False,
                 inv_link: Optional[Callable] = None):

        self.inv_link = inv_link

        # process covariance:
        self._dynamic_state_elements = []
        if process_variance:
            self._dynamic_state_elements = covariates if isinstance(process_variance, bool) else process_variance

        super().__init__(id=id, state_elements=covariates)
        for cov in covariates:
            self._set_transition(from_element=cov, to_element=cov, value=1.0)

    @property
    def dynamic_state_elements(self) -> Sequence[str]:
        return self._dynamic_state_elements

    def param_dict(self) -> torch.nn.ParameterDict:
        return torch.nn.ParameterDict()  # no parameters

    def add_measure(self, measure: str) -> 'LinearModel':
        for cov in self.state_elements:
            self._set_measure(measure=measure, state_element=cov, value=0., inv_link=self.inv_link)
        return self

    # noinspection PyMethodOverriding
    def for_batch(self, num_groups: int, num_timesteps: int, lm_input: Tensor) -> ProcessForBatch:
        for_batch = super().for_batch(num_groups=num_groups, num_timesteps=num_timesteps)

        if not isinstance(lm_input, Tensor):
            raise ValueError(f"Process {self.id} received 'lm_input' that is not a Tensor.")
        elif lm_input.requires_grad:
            raise ValueError(f"Process {self.id} received 'lm_input' that requires_grad, which is not allowed.")
        elif torch.isnan(lm_input).any():
            raise ValueError(f"Process {self.id} received 'lm_input' that has nans.")

        num_states = len(self.state_elements)
        mm_num_groups, mm_num_ts, mm_num_covs = lm_input.shape
        assert mm_num_groups == num_groups, f"Batch-size is {num_groups}, but lm_input.shape[0] is {mm_num_groups}."
        assert mm_num_ts == num_timesteps, f"Batch num. timesteps is {num_timesteps}, but lm_input.shape[1] is {mm_num_ts}."
        assert mm_num_covs == num_states, f"Expected {num_states} covariates, but lm_input.shape[2] = {mm_num_covs}."

        for measure in self.measures:
            for i, cov in enumerate(self.state_elements):
                for_batch.adjust_measure(measure=measure, state_element=cov, adjustment=lm_input[:, :, i])

        return for_batch
