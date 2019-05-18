import importlib
from collections import namedtuple
from math import pi, sqrt
from typing import Tuple, Optional, Sequence, Dict

import torch

from torch import Tensor

import numpy as np

std_normal = torch.distributions.Normal(0, 1)


class Cens(namedtuple('Cens', field_names=['obs', 'lower', 'upper'], defaults=[None, None])):
    def _for_fill(self, x):
        return x is None or isinstance(x, (int, float))

    def to_array(self) -> np.ndarray:
        obs = self._standardize_array(self.obs)

        stack = getattr(importlib.import_module(type(obs).__module__), 'stack')
        full_like = getattr(importlib.import_module(type(obs).__module__), 'full_like')

        if len(obs.shape) != 1:
            raise RuntimeError("Cannot convert to array unless len(self.obs.shape) is 1.")

        if self._for_fill(self.lower):
            lower = full_like(obs, -float('inf') if self.lower is None else self.lower)
        else:
            lower = self._standardize_array(self.lower)
        if obs.shape != lower.shape:
            raise RuntimeError("obs.shape != lower.shape")

        if self._for_fill(self.upper):
            upper = full_like(obs, float('inf') if self.upper is None else self.upper)
        else:
            upper = self._standardize_array(self.upper)
        if obs.shape != upper.shape:
            raise RuntimeError("obs.shape != upper.shape")

        if (lower == upper).any():
            raise RuntimeError("lower cannot == upper")

        arr = stack([obs, lower, upper], 1)
        return arr

    @staticmethod
    def _standardize_array(x):
        if not isinstance(x, (torch.Tensor, np.ndarray)) and isinstance(getattr(x, 'values', None), np.ndarray):
            return x.values
        return x


def tobit_adjustment(mean: Tensor,
                     cov: Tensor,
                     lower: Optional[Tensor] = None,
                     upper: Optional[Tensor] = None,
                     probs: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
    if upper is None:
        upper = torch.full_like(mean, float('inf'))
    if lower is None:
        lower = torch.full_like(mean, -float('inf'))

    is_cens_up = torch.isfinite(upper)
    is_cens_lo = torch.isfinite(lower)

    if not is_cens_up.any() and not is_cens_lo.any():
        return mean, cov

    F1, F2 = _F1F2(mean, cov, lower, upper)

    std = torch.diagonal(cov, dim1=-2, dim2=-1).sqrt()
    sqrt_pi = pi ** .5

    # prob censoring:
    if probs is None:
        prob_lo, prob_up = tobit_probs(mean=mean,
                                       cov=cov,
                                       lower=lower,
                                       upper=upper)
    else:
        prob_lo, prob_up = probs

    # adjust mean:
    lower_adj = torch.zeros_like(mean)
    lower_adj[is_cens_lo] = prob_lo[is_cens_lo] * lower[is_cens_lo]
    upper_adj = torch.zeros_like(mean)
    upper_adj[is_cens_up] = prob_up[is_cens_up] * upper[is_cens_up]
    mean_if_uncens = mean + (sqrt(2. / pi) * F1) * std
    mean_uncens_adj = (1. - prob_up - prob_lo) * mean_if_uncens
    mean_adj = mean_uncens_adj + upper_adj + lower_adj

    # adjust cov:
    diag_adj = torch.zeros_like(mean)
    for m in range(mean.shape[-1]):
        diag_adj[..., m] = (1. + 2. / sqrt_pi * F2[..., m] - 2. / pi * (F1[..., m] ** 2)) * cov[..., m, m]

    cov_adj = torch.diag_embed(diag_adj)

    return mean_adj, cov_adj


def tobit_probs(mean: Tensor,
                cov: Tensor,
                lower: Optional[Tensor] = None,
                upper: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    # CDF not well behaved at tails, truncate
    clamp = lambda z: torch.clamp(z, -5., 5.)

    if upper is None:
        upper = torch.empty_like(mean)
        upper[:] = float('inf')
    if lower is None:
        lower = torch.empty_like(mean)
        lower[:] = float('-inf')

    std = torch.diagonal(cov, dim1=-2, dim2=-1)
    probs_up = torch.zeros_like(mean)
    is_cens_up = torch.isfinite(upper)
    upper_z = (upper[is_cens_up] - mean[is_cens_up]) / std[is_cens_up]
    probs_up[is_cens_up] = 1. - std_normal.cdf(clamp(upper_z))

    probs_lo = torch.zeros_like(mean)
    is_cens_lo = torch.isfinite(lower)
    lower_z = (lower[is_cens_lo] - mean[is_cens_lo]) / std[is_cens_lo]
    probs_lo[is_cens_lo] = std_normal.cdf(clamp(lower_z))

    return probs_lo, probs_up


def erfcx(x: Tensor) -> Tensor:
    """M. M. Shepherd and J. G. Laframboise,
       MATHEMATICS OF COMPUTATION 36, 249 (1981)
    """

    K = 3.75
    y = (torch.abs(x) - K) / (torch.abs(x) + K)
    y2 = 2.0 * y
    (d, dd) = (-0.4e-20, 0.0)
    (d, dd) = (y2 * d - dd + 0.3e-20, d)
    (d, dd) = (y2 * d - dd + 0.97e-19, d)
    (d, dd) = (y2 * d - dd + 0.27e-19, d)
    (d, dd) = (y2 * d - dd + -0.2187e-17, d)
    (d, dd) = (y2 * d - dd + -0.2237e-17, d)
    (d, dd) = (y2 * d - dd + 0.50681e-16, d)
    (d, dd) = (y2 * d - dd + 0.74182e-16, d)
    (d, dd) = (y2 * d - dd + -0.1250795e-14, d)
    (d, dd) = (y2 * d - dd + -0.1864563e-14, d)
    (d, dd) = (y2 * d - dd + 0.33478119e-13, d)
    (d, dd) = (y2 * d - dd + 0.32525481e-13, d)
    (d, dd) = (y2 * d - dd + -0.965469675e-12, d)
    (d, dd) = (y2 * d - dd + 0.194558685e-12, d)
    (d, dd) = (y2 * d - dd + 0.28687950109e-10, d)
    (d, dd) = (y2 * d - dd + -0.63180883409e-10, d)
    (d, dd) = (y2 * d - dd + -0.775440020883e-09, d)
    (d, dd) = (y2 * d - dd + 0.4521959811218e-08, d)
    (d, dd) = (y2 * d - dd + 0.10764999465671e-07, d)
    (d, dd) = (y2 * d - dd + -0.218864010492344e-06, d)
    (d, dd) = (y2 * d - dd + 0.774038306619849e-06, d)
    (d, dd) = (y2 * d - dd + 0.4139027986073010e-05, d)
    (d, dd) = (y2 * d - dd + -0.69169733025012064e-04, d)
    (d, dd) = (y2 * d - dd + 0.490775836525808632e-03, d)
    (d, dd) = (y2 * d - dd + -0.2413163540417608191e-02, d)
    (d, dd) = (y2 * d - dd + 0.9074997670705265094e-02, d)
    (d, dd) = (y2 * d - dd + -0.26658668435305752277e-01, d)
    (d, dd) = (y2 * d - dd + 0.59209939998191890498e-01, d)
    (d, dd) = (y2 * d - dd + -0.84249133366517915584e-01, d)
    (d, dd) = (y2 * d - dd + -0.4590054580646477331e-02, d)
    d = y * d - dd + 0.1177578934567401754080e+01

    result = d / (1.0 + 2.0 * torch.abs(x))
    x_neg = (x < 0)
    result[x_neg] = 2.0 * torch.exp(x[x_neg] ** 2) - result[x_neg]

    return result


def _F1F2_no_inf(x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
    if (x.abs() > 3).any() or (y.abs() > 3).any():
        raise RuntimeError("_F1F2_no_inf not stable for inputs with abs(value) > 3")

    numer_1 = torch.exp(-x ** 2) - torch.exp(-y ** 2)
    numer_2 = x * torch.exp(-x ** 2) - y * torch.exp(-y ** 2)
    denom = torch.erf(y) - torch.erf(x)

    F1 = numer_1 / denom
    F2 = numer_2 / denom
    return F1, F2


def _F1F2(mean: Tensor,
          cov: Tensor,
          lower: Tensor,
          upper: Tensor) -> Tuple[Tensor, Tensor]:
    is_cens_up = torch.isfinite(upper)
    is_cens_lo = torch.isfinite(lower)

    std = torch.diagonal(cov, dim1=-2, dim2=-1).sqrt()

    # mask out the infs before any gradients are being tracked:
    alpha = torch.zeros_like(mean)
    alpha[is_cens_lo] = (lower[is_cens_lo] - mean[is_cens_lo]) / std[is_cens_lo]
    beta = torch.zeros_like(mean)
    beta[is_cens_up] = (upper[is_cens_up] - mean[is_cens_up]) / std[is_cens_up]

    # _F1F2_no_inf unstable for large z-scores, so use the lim(+/-inf) version for those as well
    is_cens_up = is_cens_up & (beta.data < 4.)
    is_cens_lo = is_cens_lo & (alpha.data > -4.)
    is_cens_both = is_cens_up & is_cens_lo

    #
    sqrt_2 = 2. ** .5
    x = alpha / sqrt_2
    y = beta / sqrt_2

    # uncensored
    F1, F2 = torch.zeros_like(mean), torch.zeros_like(mean)

    # censored both:
    F1[is_cens_both], F2[is_cens_both] = _F1F2_no_inf(x[is_cens_both], y[is_cens_both])

    # censored lower, uncensored upper:
    F1[is_cens_lo & ~is_cens_up] = 1. / erfcx(x[is_cens_lo & ~is_cens_up])
    F2[is_cens_lo & ~is_cens_up] = x[is_cens_lo & ~is_cens_up] / erfcx(x[is_cens_lo & ~is_cens_up])

    # uncensored lower, censored upper:
    F1[~is_cens_lo & is_cens_up] = -1. / erfcx(-y[~is_cens_lo & is_cens_up])
    F2[~is_cens_lo & is_cens_up] = -y[~is_cens_lo & is_cens_up] / erfcx(-y[~is_cens_lo & is_cens_up])

    return F1, F2
