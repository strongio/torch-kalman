from typing import Generator, Tuple, Sequence, Optional

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_kalman.covariance import Covariance
from torch_kalman.process import Process
from torch_kalman.process.for_batch import ProcessForBatch
from torch_kalman.utils import itervalues_sorted_keys


class HLM(Process):
    def __init__(self,
                 id: str,
                 covariates: Sequence[str],
                 model_mat_kwarg_name: Optional[str] = None):
        # transitions:
        transitions = {covariate: {covariate: 1.0} for covariate in covariates}

        # initial state:
        ns = len(covariates)
        self.initial_state_mean_params = Parameter(torch.randn(ns))
        self.initial_state_cov_params = dict(log_diag=Parameter(data=torch.randn(ns)),
                                             off_diag=Parameter(data=torch.randn(int(ns * (ns - 1) / 2))))

        # super:
        super().__init__(id=id, state_elements=covariates, transitions=transitions)

        # expected kwargs
        model_mat_kwarg_name = model_mat_kwarg_name or id  # use the id if they didn't specify
        self.expected_batch_kwargs = (model_mat_kwarg_name,)

    # noinspection PyMethodOverriding
    def add_measure(self, measure: str) -> None:
        for state_element in self.state_elements:
            self.state_elements_to_measures[(measure, state_element)] = None

    def initial_state(self, **kwargs) -> Tuple[Tensor, Tensor]:
        means = self.initial_state_mean_params
        covs = Covariance.from_log_cholesky(**self.initial_state_cov_params, device=self.device)
        return means, covs

    def parameters(self) -> Generator[Parameter, None, None]:
        yield self.initial_state_mean_params
        for param in itervalues_sorted_keys(self.initial_state_cov_params):
            yield param

    def covariance(self) -> Covariance:
        ns = len(self.state_elements)
        out = torch.empty(size=(ns, ns), device=self.device)
        out[:, :] = 0.
        return out

    def for_batch(self, input: Tensor, **kwargs) -> ProcessForBatch:
        assert self.state_elements_to_measures, f"HLM process '{self.id}' has no measures."

        re_model_mat = kwargs.get(self.expected_batch_kwargs[0], None)

        if re_model_mat is None:
            raise ValueError(f"Required argument `{self.expected_batch_kwargs[0]}` not found.")
        elif torch.isnan(re_model_mat).any():
            raise ValueError(f"nans not allowed in `{self.expected_batch_kwargs[0]}` tensor")

        for_batch = super().for_batch(input)

        for i, covariate in enumerate(self.state_elements):
            for measure in self.measures():
                for_batch.add_measure(measure=measure, state_element=covariate, values=re_model_mat[:, :, i])

        return for_batch
