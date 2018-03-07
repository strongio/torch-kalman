from torch import stack


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
    return stack([x[i].t() for i in xrange(ns)], 0)
