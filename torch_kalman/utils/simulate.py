from typing import Optional

from torch import Tensor
from torch.distributions import MultivariateNormal
from torch.distributions.multivariate_normal import _batch_mv
from torch.distributions.utils import _standard_normal


def deterministic_sample_mvnorm(distribution: MultivariateNormal, eps: Optional[Tensor] = None) -> Tensor:
    if isinstance(eps, Tensor):
        if eps.shape[-len(distribution.event_shape):] != distribution.event_shape:
            raise RuntimeError(f"Expected shape ending in {distribution.event_shape}, got {eps.shape}.")
    else:
        shape = distribution.batch_shape + distribution.event_shape
        if eps is None:
            eps = 1.0
        eps *= _standard_normal(shape, dtype=distribution.loc.dtype, device=distribution.loc.device)
    return distribution.loc + _batch_mv(distribution._unbroadcasted_scale_tril, eps)
