import torch


def log_std_to_var(log_std):
    """
    :param log_std: The log(standard-deviation)
    :return: The variance
    """
    return torch.pow(torch.exp(log_std), 2)


def expand(x, ns):
    """
    Expand a tensor of some shape into a new dimension, repeating its `ns` times.

    :param x: Tensor
    :param ns: Number of repeats
    :return: Tensor repeated into new dimension.
    """
    return x.expand(ns, *x.data.shape)


def batch_transpose(x):
    """
    Given a tensor whose first dimension is the batch, transpose each batch.

    :param x: A tensor whose first dimension is the batch.
    :return: x transposed batchwise.
    """
    ns = x.data.shape[0]
    return torch.stack([x[i].t() for i in xrange(ns)], 0)
