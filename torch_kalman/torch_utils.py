import torch
from torch import Tensor


def batch_inverse(x: Tensor) -> Tensor:
    # please keep track of https://github.com/pytorch/pytorch/pull/9102
    return torch.cat([torch.inverse(x[i, :, :]).unsqueeze(0) for i in range(len(x))], 0)
