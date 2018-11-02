from typing import Generator, Optional, Tuple, Dict, Set

import numpy as np
import torch

from torch import Tensor

from torch.nn import Parameter

from torch_kalman.covariance import Covariance
from torch_kalman.process.for_batch import ProcessForBatch
from torch_kalman.process.processes.season.base import DateAware
from torch_kalman.utils import fourier_series, itervalues_sorted_keys

from IPython.core.debugger import Pdb


def zpad(x, n):
    return str(x).rjust(n, "0")


class NestedSeason(DateAware):
    """
    Defined as multiple smooth (fourier) seasonal patterns that themselves are swapped out depending on a discrete seasonal
    process.

    For example, each day-of-the-week follows a different seasonal pattern over the course of the year.

    Or, the hours in a day follow a pattern, but the pattern is different for each day of the week.
    """

    def __init__(self,
                 id: str,
                 seasonal_period: float,
                 K: int,
                 discrete_season_id: str,
                 **kwargs):

        #
        self.discrete_season_id = discrete_season_id
        self.K = K
        self.seasonal_period = seasonal_period

        # parameters:
        # (note degrees of freedom would be same if this were just one fourier -- i.e., they all share cov and initial state)
        # initial-values
        ns = self.K * 2
        self.initial_state_mean_params = Parameter(torch.randn(ns))
        self.initial_state_cov_params = dict(log_diag=Parameter(data=torch.randn(ns)),
                                             off_diag=Parameter(data=torch.randn(int(ns * (ns - 1) / 2))))
        # process covariance:
        self.cov_cholesky_log_diag = Parameter(data=torch.zeros(ns))
        self.cov_cholesky_off_diag = Parameter(data=torch.zeros(int(ns * (ns - 1) / 2)))

        # these are created when `link_to_design` is called:
        self.get_discrete_season = None
        self.discrete_seasonal_period = False
        self.measures_queue = set()

        # noinspection PyTypeChecker
        super().__init__(id=id, state_elements=None, transitions=None, **kwargs)

        # writing measure-matrix is slow, no need to do it repeatedly:
        self.measure_cache = {}

    @property
    def transitions(self):
        if self._transitions is None:
            raise RuntimeError(f"Cannot access `transitions` until process {self.id} has been linked to a design.")
        return super().transitions

    @property
    def state_elements(self):
        if self._state_elements is None:
            raise RuntimeError(f"Cannot access `state_elements` until process {self.id} has been linked to a design.")
        return super().state_elements

    def link_to_design(self, design: 'Design'):
        super().link_to_design(design)

        assert not self._state_elements, f"`link_to_design` has already been called on process {self.id}"

        self.discrete_seasonal_period = design.processes[self.discrete_season_id].seasonal_period
        self.get_discrete_season = design.processes[self.discrete_season_id].get_season

        state_elements = []
        transitions = {}
        pad_n = len(str(self.discrete_seasonal_period))
        for r in range(self.K):
            for c in range(2):
                st_els = [f"{r},{c},{zpad(i, pad_n)}" for i in range(self.discrete_seasonal_period)]
                state_elements.extend(st_els)
                for st_el in st_els:
                    transitions[st_el] = {st_el: 1.0}

        self._state_elements = state_elements
        self._transitions = transitions

        for measure in self.measures_queue:
            for state_element in self.state_elements:
                super().add_measure(measure=measure, state_element=state_element, value=None)

        self.validate_state_elements(state_elements=self._state_elements, transitions=self._transitions)

    def for_batch(self,
                  batch_size: int,
                  time: Optional[int] = None,
                  start_datetimes: Optional[np.ndarray] = None,
                  cache: bool = True
                  ) -> ProcessForBatch:

        # super:
        for_batch = super().for_batch(batch_size=batch_size)

        # determine the delta (integer time accounting for different groups having different start datetimes)
        delta = self.get_delta(batch_size=batch_size, time=time, start_datetimes=start_datetimes)

        # determine which season we are in:
        discrete_season = self.get_discrete_season(delta)
        smooth_season = delta % self.seasonal_period

        # determine measurement function:
        assert not for_batch.batch_ses_to_measures, "Please report this error to the package maintainer."
        if cache:
            key = discrete_season.tostring(), smooth_season.tostring()
            if key not in self.measure_cache.keys():
                self.measure_cache[key] = self.make_batch_measures(discrete_season, smooth_season)
            for_batch.batch_ses_to_measures = self.measure_cache[key]
        else:
            for_batch.batch_ses_to_measures = self.make_batch_measures(discrete_season, smooth_season)

        return for_batch

    def make_batch_measures(self, discrete_season: np.ndarray, smooth_season: np.ndarray) -> Dict[Tuple[str, str], Tensor]:
        for_batch = super().for_batch(batch_size=len(discrete_season))

        # generate the fourier matrix:
        fourier_mat = fourier_series(time=Tensor(smooth_season), seasonal_period=self.seasonal_period, K=self.K)

        # for each state-element, use fourier values only if we are in the discrete-season (se_discrete_season)
        for measure in self.measures():
            for state_element in self.state_elements:
                r, c, se_discrete_season = (int(x) for x in state_element.split(sep=","))
                is_measured = Tensor((se_discrete_season == discrete_season).astype('float32'), device=self.device)
                values = fourier_mat[:, r, c] * is_measured
                for_batch.add_measure(measure=measure, state_element=state_element, values=values)

        return for_batch.batch_ses_to_measures

    # noinspection PyMethodOverriding
    def add_measure(self, measure: str) -> None:
        self.measures_queue.add(measure)

    def measures(self) -> Set[str]:
        if self._state_elements is None:
            raise RuntimeError("Cannot access `measures` until `link_to_design` is called.")
        return super().measures()

    def parameters(self) -> Generator[Parameter, None, None]:
        yield self.cov_cholesky_log_diag
        yield self.cov_cholesky_off_diag
        yield self.initial_state_mean_params
        for param in itervalues_sorted_keys(self.initial_state_cov_params):
            yield param

    def block_diag_cov(self, block: Covariance) -> Covariance:
        # create empty covariance matrix:
        state_size = len(self.state_elements)
        cov = Covariance(size=(state_size, state_size), device=self.device)
        cov[:] = 0.

        # fill blockwise:
        start = 0
        for season in range(self.discrete_seasonal_period):
            end = start + block.shape[1]
            cov[np.ix_(range(start, end), range(start, end))] = block
            start = end

        return cov

    def initial_state(self, batch_size: int, **kwargs) -> Tuple[Tensor, Tensor]:

        # means:
        means = self.initial_state_mean_params.repeat(self.discrete_seasonal_period).expand(batch_size, -1)

        # covs:
        block = Covariance.from_log_cholesky(**self.initial_state_cov_params, device=self.device)
        covs = self.block_diag_cov(block).expand(batch_size, -1, -1)
        return means, covs

    def covariance(self) -> Covariance:
        block = Covariance.from_log_cholesky(log_diag=self.cov_cholesky_log_diag,
                                             off_diag=self.cov_cholesky_off_diag,
                                             device=self.device)
        return self.block_diag_cov(block)

