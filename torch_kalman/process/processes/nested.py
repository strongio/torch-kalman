from typing import Generator, Tuple

import torch

from torch import Tensor
from torch.nn import Parameter

from torch_kalman.covariance import Covariance
from torch_kalman.process import Process
from torch_kalman.utils import dict_key_replace, zpad

import numpy as np


class Nested(Process):
    """
    Allows you to nest one process inside of a discrete seasonal process, so that the sub-process has a different state for
    each season. For example, each day-of-the-week follows a different fourier pattern over the course of the year; or, the
    hours in a day follow a pattern, but the pattern is different for each day of the week; or, the way the weather affects
    the measurement is different for each day of the week.
    """

    def __init__(self,
                 id: str,
                 discrete_season_id: str,
                 process: Process):
        #
        self.discrete_season_id = discrete_season_id

        self.sub_process = process

        # these are created when `link_to_design` is called:
        self.discrete_process = None

        # noinspection PyTypeChecker
        super().__init__(id=id, state_elements=None, transitions=None)

        # writing measure-matrix is slow, no need to do it repeatedly:
        self.measure_cache = {}

        self.expected_batch_kwargs = list(self.sub_process.expected_batch_kwargs)

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

    def sub_state_element_rename(self, sub_el: str, season: int) -> str:
        pad_n = len(str(self.discrete_process.seasonal_period))
        return f"{sub_el}__{zpad(season, pad_n)}"

    def for_batch(self, batch_size: int, **kwargs) -> 'ProcessForBatch':

        for_batch = super().for_batch(batch_size=batch_size)
        sub_for_batch = self.sub_process.for_batch(batch_size=batch_size, **kwargs)

        discrete_season = self.discrete_process.get_season(batch_size=batch_size,
                                                           time=kwargs['time'],
                                                           start_datetimes=kwargs.get('start_datetimes', None))

        for (measure, sub_state_element), values in sub_for_batch.batch_ses_to_measures.items():
            for i in range(self.discrete_process.seasonal_period):
                state_element = self.sub_state_element_rename(sub_state_element, i)
                is_measured = Tensor((i == discrete_season).astype('float32'), device=self.device)
                for_batch.add_measure(measure=measure, state_element=state_element, values=values * is_measured)

        return for_batch

    def link_to_design(self, design: 'Design'):
        super().link_to_design(design)
        self.sub_process.link_to_design(design)

        assert not self._state_elements, f"`link_to_design` has already been called on process {self.id}"

        self.discrete_process = design.processes[self.discrete_season_id]
        for kwarg in self.discrete_process.expected_batch_kwargs:
            if kwarg not in self.expected_batch_kwargs:
                self.expected_batch_kwargs.append(kwarg)

        state_elements = []
        transitions = {}
        for i in range(self.discrete_process.seasonal_period):
            sub_transitions = self.sub_process.transitions.copy()
            for sub_el in self.sub_process.state_elements:
                new_el = self.sub_state_element_rename(sub_el, i)
                sub_transitions = dict_key_replace(sub_transitions, sub_el, new_el)
                state_elements.append(new_el)
            transitions.update(sub_transitions)

        self._state_elements = state_elements
        self._transitions = transitions

        for measure in self.sub_process.measures():
            for state_element in self.state_elements:
                super().add_measure(measure=measure, state_element=state_element, value=None)

        self.validate_state_elements(state_elements=self._state_elements, transitions=self._transitions)

    # noinspection PyMethodOverriding
    def add_measure(self, *args, **kwargs) -> None:
        self.sub_process.add_measure(*args, **kwargs)

    def initial_state(self, batch_size: int, **kwargs) -> Tuple[Tensor, Tensor]:

        ns = len(self.state_elements)

        means = torch.zeros((batch_size, ns))
        covs = torch.zeros((batch_size, ns, ns))

        start = 0
        for _ in range(self.discrete_process.seasonal_period):
            submeans, subcovs = self.sub_process.initial_state(batch_size=batch_size, **kwargs)
            end = start + len(self.sub_process.state_elements)

            # means:
            means[:, start:end] = submeans

            # covs:
            covs[np.ix_(range(batch_size), range(start, end), range(start, end))] = subcovs

            start = end

        return means, covs

    def parameters(self) -> Generator[Parameter, None, None]:
        yield from self.sub_process.parameters()

    def covariance(self) -> Covariance:
        ns = len(self.state_elements)
        cov = torch.zeros((ns, ns))
        subcov = self.sub_process.covariance()

        start = 0
        for _ in range(self.discrete_process.seasonal_period):
            end = start + len(self.sub_process.state_elements)
            cov[np.ix_(range(start, end), range(start, end))] = subcov
            start = end

        return cov
