from typing import List, Optional, Sequence

import torch

from torch import jit, Tensor, nn

from torch_kalman.rewrite.helpers import ReturnValues, SingleOutput, Exp


class Covariance(jit.ScriptModule):
    def __init__(self,
                 rank: int,
                 empty_idx: List[int] = (),
                 method: str = 'log_cholesky'):
        super(Covariance, self).__init__()
        self.rank = rank
        if len(empty_idx) == 0:
            empty_idx = [self.rank + 1]  # jit doesn't seem to like empty lists
        self.empty_idx = empty_idx
        self.method = method
        if self.method == 'log_cholesky':
            self.chol_params = nn.ModuleList()
            for r in range(self.rank):
                for c in range(self.rank):
                    if r in self.empty_idx or c in self.empty_idx:
                        self.chol_params.append(ReturnValues(torch.zeros(1)))
                    elif r == c:
                        self.chol_params.append(SingleOutput(transform=Exp()))
                    elif r > c:
                        self.chol_params.append(SingleOutput())
                    else:
                        self.chol_params.append(ReturnValues(torch.zeros(1)))
        else:
            raise NotImplementedError(method)

    @jit.script_method
    def forward(self, inputs: List[Tensor]) -> Tensor:
        if self.method == 'log_cholesky':
            num_groups = inputs[0].shape[0]
            cov = self.from_log_cholesky()
            # TODO: predicting diag-multi?
            return cov.expand(num_groups, -1, -1)
        else:
            raise NotImplementedError(self.method)

    def from_log_cholesky(self):
        L = torch.jit.annotate(List[Tensor], [])
        for i, module in enumerate(self.chol_params):
            L += [module()]
        L = torch.stack(L).view(self.rank, self.rank)
        return L @ L.t()
