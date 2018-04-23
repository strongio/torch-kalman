import torch
from torch.autograd import Variable
from torch.nn import Parameter


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
    if isinstance(x, Variable):
        return x.expand(ns, *x.data.shape)
    else:
        return x.expand(ns, *x.shape)


def batch_diag(x):
    """
    Interpreting a 3D tensor as a list of matrices with the first dimension indexing the elements in list, return the diag of
     each matrix.
    :param x: A 3D tensor.
    :return: A 2D tensor, the first dimension that of x.
    """
    n = x.data.shape[0]
    return torch.cat([torch.diag(x[i])[None, :] for i in range(n)], 1)


def quad_form_diag(std_devs, corr_mat):
    """
    Generate a covariance matrix from marginal std-devs and a correlation matrix.

    :param std_devs: A list of Variables or a 1D variable, with the std-deviations.
    :param corr_mat: A correlation matrix Variable
    :return: A covariance matrix
    """
    n = len(std_devs)
    variance_diag = Variable(torch.zeros((n, n)))
    for i in range(n):
        variance_diag[i, i] = torch.pow(std_devs[i], 2)
    return torch.mm(torch.mm(variance_diag, corr_mat), variance_diag)


def Param0(*sizes):
    if len(sizes) == 0:
        sizes = (1,)
    return Parameter(torch.zeros(*sizes))


def ParamRand(*sizes):
    if len(sizes) == 0:
        sizes = (1,)
    return Parameter(torch.randn(*sizes))
