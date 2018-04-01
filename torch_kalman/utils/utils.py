import numpy as np


def cummean(A, axis):
    """
    Take the cumulative mean along an axis.

    :param A: An array
    :param axis: The axis.
    :return: Array with same shape as A.
    """
    return np.true_divide(A.cumsum(axis), np.arange(1, A.shape[axis] + 1))


def train_val_split(array, prop=.75):
    """

    :param array: Split an array whose first dimension corresponds to batch into training and validation.
    :param prop: The proportion going to the training set.
    :return: A tuple of arrays.
    """
    train_idx = np.random.choice(array.shape[0], int(round(array.shape[0] * prop)), replace=False)
    all_idx = np.arange(array.shape[0])
    return array[np.in1d(all_idx, train_idx)], array[~np.in1d(all_idx, train_idx)]


def make_callable(x):
    """
    :param x: An object
    :return: If x is callable, then x. Otherwise, a function that returns x.
    """
    if callable(x):
        return x
    else:
        return lambda: x
