import torch


def batch_inverse(x):
    # please keep track of https://github.com/pytorch/pytorch/pull/9102
    return torch.cat([torch.inverse(x[i, :, :]).unsqueeze(0) for i in range(len(x))], 0)