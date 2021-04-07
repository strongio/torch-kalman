"""
The :class:`.KalmanFilter` is a :class:`torch.nn.Module` which generates forecasts, in the form of
:class:`.Predictions` objects, which can be used for training (:func:`~torchcast.kalman_filter.Predictions.log_prob()`)
, evaluation (:func:`~torchcast.kalman_filter.Predictions.to_dataframe()`) or visualization
(:func:`~torchcast.kalman_filter.Predictions.plot()`).

----------
"""

from .base import KalmanFilter
from .predictions import Predictions
