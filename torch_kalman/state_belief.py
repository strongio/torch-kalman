# noinspection PyPep8Naming
from typing import Tuple

import torch
from torch import Tensor


class StateBelief:
    def __init__(self, mean: Tensor, cov: Tensor):
        self.mean = mean
        self.cov = cov

    def to_tensors(self) -> Tuple[Tensor, Tensor]:
        return self.mean, self.cov

    def predict(self, F: Tensor, Q: Tensor):
        raise NotImplementedError

    def update(self, obs: Tensor, H: Tensor, R: Tensor):
        raise NotImplementedError


class Gaussian(StateBelief):
    def predict(self, F: Tensor, Q: Tensor) -> StateBelief:
        Ft = F.permute(0, 2, 1)
        mean = torch.bmm(F, self.mean)
        cov = torch.bmm(torch.bmm(F, self.cov), Ft) + Q
        return self.__class__(mean=mean, cov=cov)

    def update(self, obs: Tensor, H: Tensor, R: Tensor) -> StateBelief:
        raise NotImplementedError
