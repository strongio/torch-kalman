"""
`Process` modules are used to specify the latent-states (and their temporal dynamics) underlying your time-series(es):

* :class:`.LocalLevel` - a random-walk
* :class:`.LocalTrend` - a random-walk with (optionally damped) velocity
* :class:`.Season` - a process with seasonal structure, implementing the fourier-series based model from
  `De Livera, A.M., Hyndman, R.J., & Snyder, R. D. (2011)`
* :class:`.LinearModel` - a simple linear-model allowing for external predictors
* :class:`.NN` - an arbitrary |nn.Module| is used to translate external predictors into latent states.

----------
"""

from .regression import NN, LinearModel
from .local import LocalLevel, LocalTrend
from .season import DiscreteSeason, Season, TBATS