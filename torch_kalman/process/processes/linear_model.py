from typing import Sequence, Optional, Union, Callable, Collection

import torch

from torch import Tensor

from torch_kalman.process import Process
from torch_kalman.process.mixins.has_predictors import HasPredictors
from torch_kalman.internals.utils import split_flat


class LinearModel(HasPredictors, Process):

    def __init__(self,
                 id: str,
                 covariates: Sequence[str],
                 process_variance: Union[bool, Collection[str]] = False,
                 init_variance: Union[bool, Collection[str]] = True,
                 inv_link: Optional[Callable] = None):
        if isinstance(covariates, str):
            raise TypeError("`covariates` should be sequence of strings, not single string")

        self.inv_link = inv_link

        # process covariance:
        self._dynamic_state_elements = []
        if process_variance:
            self._dynamic_state_elements = covariates if isinstance(process_variance, bool) else process_variance
            extras = set(self._dynamic_state_elements) - set(covariates)
            if len(extras):
                raise ValueError(f"`process_variance` includes items not in `covariates`:{extras}")

        # initial covariance:
        self._fixed_state_elements = []
        if init_variance is False:
            init_variance = []
        if init_variance is not True:
            extras = set(init_variance) - set(covariates)
            if len(extras):
                raise ValueError(f"`init_variance` includes items not in `covariates`:{extras}")
            self._fixed_state_elements = [cov for cov in covariates if cov not in init_variance]

        super().__init__(id=id, state_elements=covariates)

        for cov in covariates:
            self._set_transition(from_element=cov, to_element=cov, value=1.0)

    @property
    def dynamic_state_elements(self) -> Sequence[str]:
        return self._dynamic_state_elements

    @property
    def fixed_state_elements(self):
        return self._fixed_state_elements

    def param_dict(self) -> torch.nn.ParameterDict:
        return torch.nn.ParameterDict()  # no parameters

    def add_measure(self, measure: str) -> 'LinearModel':
        for cov in self.state_elements:
            self._set_measure(measure=measure, state_element=cov, value=0., ilink=self.inv_link)
        return self

    def for_batch(self,
                  num_groups: int,
                  num_timesteps: int,
                  predictors: Tensor,
                  allow_extra_timesteps: bool = False) -> 'LinearModel':
        
        for_batch = super().for_batch(
            num_groups=num_groups,
            num_timesteps=num_timesteps,
            predictors=predictors,
            expected_num_predictors=len(self.state_elements),
            allow_extra_timesteps=allow_extra_timesteps
        )

        if predictors.shape[1] > num_timesteps:
            predictors = predictors[:, 0:num_timesteps, :]

        for measure in self.measures:
            for i, cov in enumerate(self.state_elements):
                for_batch._adjust_measure(
                    measure=measure,
                    state_element=cov,
                    adjustment=split_flat(predictors[:, :, i], dim=1)
                )

        return for_batch
