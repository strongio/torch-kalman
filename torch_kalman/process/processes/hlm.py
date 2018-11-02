from typing import Generator, Tuple, Sequence, Optional

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_kalman.covariance import Covariance
from torch_kalman.process import Process
from torch_kalman.process.for_batch import ProcessForBatch


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
        self.expected_batch_kwargs = ('time', model_mat_kwarg_name)

    # noinspection PyMethodOverriding
    def add_measure(self, measure: str) -> None:
        for state_element in self.state_elements:
            self.state_elements_to_measures[(measure, state_element)] = None

    def initial_state(self, batch_size: int, **kwargs) -> Tuple[Tensor, Tensor]:
        means = self.initial_state_mean_params.expand(batch_size, -1)
        covs = Covariance.from_log_cholesky(**self.initial_state_cov_params, device=self.device).expand(batch_size, -1, -1)
        return means, covs

    def parameters(self) -> Generator[Parameter, None, None]:
        yield self.initial_state_mean_params
        for param in self.initial_state_cov_params.values():
            yield param

    def covariance(self) -> Covariance:
        ns = len(self.state_elements)
        out = Covariance(size=(ns, ns))
        out[:, :] = 0.
        return out

    def for_batch(self, batch_size: int, time=None, **kwargs) -> ProcessForBatch:
        assert self.state_elements_to_measures, f"HLM process '{self.id}' has no measures."

        re_model_mat = kwargs.get(self.expected_batch_kwargs[1], None)

        if re_model_mat is None:
            raise ValueError(f"Required argument `{self.expected_batch_kwargs[1]}` not found.")
        else:
            re_model_mat = re_model_mat[:, time, :]

        for_batch = super().for_batch(batch_size=batch_size)

        for i, covariate in enumerate(self.state_elements):
            for measure in self.measures():
                for_batch.add_measure(measure=measure, state_element=covariate, values=re_model_mat[:, i])

        return for_batch
