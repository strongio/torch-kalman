from collections import namedtuple, defaultdict
from typing import Tuple, List, Optional, Sequence, Dict, Iterable, Collection
from warnings import warn

import torch
from torch import nn, Tensor

from torch_kalman.covariance import Covariance
from torch_kalman.kalman_filter.gaussian import GaussianStep
from torch_kalman.kalman_filter.state_belief_over_time import StateBeliefOverTime
from torch_kalman.process.regression import Process

KFOutput = namedtuple('KFOutput', ['means', 'covs', 'R', 'H'])


class KalmanFilter(nn.Module):
    kf_step = GaussianStep

    def __init__(self,
                 processes: Sequence[Process],
                 measures: Sequence[str],
                 compiled: bool = True,
                 **kwargs):
        super(KalmanFilter, self).__init__()
        self._validate(processes, measures)
        self.script_module = ScriptKalmanFilter(
            kf_step=self.kf_step(),
            processes=processes,
            measures=measures,
            **kwargs
        )
        if compiled:
            self.script_module = torch.jit.script(self.script_module)

    @staticmethod
    def _validate(processes: Sequence[Process], measures: Sequence[str]):
        assert not isinstance(measures, str)
        for p in processes:
            if p.measure:
                if p.measure not in measures:
                    raise RuntimeError(f"'{p.id}' has measure {p.measure} not in `measures`.")
            else:
                if len(measures) > 1:
                    raise RuntimeError(f"Must call `set_measure()` on '{p.id}' since there are multiple measures.")
                p.set_measure(measures[0])

    def named_processes(self) -> Iterable[Tuple[str, Process]]:
        for process in self.script_module.processes:
            yield process.id, process

    def named_covariances(self) -> Iterable[Tuple[str, Covariance]]:
        return [
            ('process_covariance', self.script_module.process_covariance),
            ('measure_covariance', self.script_module.measure_covariance),
            ('initial_covariance', self.script_module.initial_covariance),
        ]

    def _parse_design_kwargs(self, **kwargs) -> Dict[str, dict]:
        static_kwargs = defaultdict(dict)
        time_varying_kwargs = defaultdict(dict)
        init_state_kwargs = defaultdict(dict)
        unused = set(kwargs)
        for submodule_nm, submodule in list(self.named_processes()) + list(self.named_covariances()):
            # TODO: init_state_kwargs
            for key in submodule.expected_kwargs:
                if key != '':
                    found_key, value = self._get_design_kwarg(submodule_nm, key, kwargs)
                    unused.discard(found_key)
                    if len(value.shape) == 3:
                        time_varying_kwargs[submodule_nm][key] = value.unbind(1)
                    else:
                        static_kwargs[submodule_nm][key] = value

        if unused and unused != {'input'}:
            warn(f"There are unused keyword arguments:\n{unused}")
        return {
            'static_kwargs': dict(static_kwargs),
            'time_varying_kwargs': dict(time_varying_kwargs),
            'init_state_kwargs': dict(init_state_kwargs)
        }

    @staticmethod
    def _get_design_kwarg(owner: str, key: str, kwargs: dict) -> Tuple[str, Tensor]:
        specific_key = f"{owner}__{key}"
        if specific_key in kwargs:
            return specific_key, kwargs[specific_key]
        else:
            return key, kwargs[key]

    def _enable_process_cache(self, enable: bool = True):
        for p in self.script_module.processes:
            p.enable_cache(enable)

    def forward(self,
                input: Tensor,
                n_step: int = 1,
                out_timesteps: Optional[int] = None,
                initial_state: Optional[Tuple[Tensor, Tensor]] = None,
                **kwargs) -> StateBeliefOverTime:
        self._enable_process_cache(True)
        try:
            means, covs, R, H = self.script_module(
                input=input,
                initial_state=initial_state,
                n_step=n_step,
                out_timesteps=out_timesteps,
                **self._parse_design_kwargs(input=input, **kwargs)
            )
        finally:
            self._enable_process_cache(False)
        return StateBeliefOverTime(means, covs, R=R, H=H, kf_step=self.kf_step)


class ScriptKalmanFilter(nn.Module):
    def __init__(self,
                 kf_step: 'GaussianStep',
                 processes: Sequence[Process],
                 measures: Sequence[str],
                 process_covariance: Optional[Covariance] = None,
                 measure_covariance: Optional[Covariance] = None,
                 initial_covariance: Optional[Covariance] = None):
        super(ScriptKalmanFilter, self).__init__()

        self.kf_step = kf_step

        # measures:
        self.measures = measures
        self.measure_to_idx = {m: i for i, m in enumerate(self.measures)}

        # processes:
        self.processes = nn.ModuleList()
        self.process_to_slice: Dict[str, Tuple[int, int]] = {}
        self.state_rank = 0
        self.no_pcov_idx = []
        for p in processes:
            self.processes.append(p)
            if not p.measure:
                raise RuntimeError(f"Must call `set_measure()` on '{p.id}'")
            self.process_to_slice[p.id] = (self.state_rank, self.state_rank + len(p.state_elements))

            for i, se in enumerate(p.no_pcov_state_elements):
                self.no_pcov_idx.append(self.state_rank + i)
            self.state_rank += len(p.state_elements)

        # covariances:
        if process_covariance is None:
            process_covariance = Covariance(rank=self.state_rank, empty_idx=self.no_pcov_idx)
        self.process_covariance = process_covariance
        if measure_covariance is None:
            measure_covariance = Covariance(rank=len(self.measures))
        self.measure_covariance = measure_covariance
        if initial_covariance is None:
            initial_covariance = Covariance(rank=self.state_rank)  # TODO: fixed state-elements; low-rank
        self.initial_covariance = initial_covariance

    def get_initial_state(self,
                          input: Tensor,
                          init_state_kwargs: Dict[str, Dict[str, Tensor]],
                          init_cov_kwargs: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        num_groups = input.shape[0]

        # initial state mean:
        mean = torch.zeros(num_groups, self.state_rank)  # TODO

        # initial cov:
        cov = self.initial_covariance(init_cov_kwargs)
        if len(cov.shape) == 2:
            cov = cov.expand(num_groups, -1, -1)
        return mean, cov

    def get_design_mats(self,
                        num_groups: int,
                        design_kwargs: Dict[str, Dict[str, Tensor]],
                        tv_kwargs: List[str] = ()) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

        F = torch.zeros((num_groups, self.state_rank, self.state_rank))
        H = torch.zeros((num_groups, len(self.measures), self.state_rank))
        for process in self.processes:
            if process.id in design_kwargs.keys():
                this_process_kwargs = design_kwargs[process.id]
            else:
                this_process_kwargs = {}
            pH, pF = process(this_process_kwargs, tv_kwargs=tv_kwargs)

            _process_slice = slice(*self.process_to_slice[process.id])
            H[:, self.measure_to_idx[process.measure], _process_slice] = pH
            F[:, _process_slice, _process_slice] = pF

        Q = self.process_covariance(design_kwargs.get('process_covariance', {}))
        if len(Q.shape) == 2:
            Q = Q.expand(num_groups, -1, -1)
        R = self.measure_covariance(design_kwargs.get('measure_covariance', {}))
        if len(R.shape) == 2:
            R = R.expand(num_groups, -1, -1)

        return F, H, Q, R

    def forward(self,
                input: Optional[Tensor],
                static_kwargs: Dict[str, Dict[str, Tensor]],
                time_varying_kwargs: Dict[str, Dict[str, List[Tensor]]],
                init_state_kwargs: Dict[str, Dict[str, Tensor]],
                n_step: int = 1,
                out_timesteps: Optional[int] = None,
                initial_state: Optional[Tuple[Tensor, Tensor]] = None) -> KFOutput:

        if initial_state is None:
            assert input is not None
            mean1step, cov1step = self.get_initial_state(
                input=input,
                init_state_kwargs=init_state_kwargs,
                init_cov_kwargs=static_kwargs.pop('initial_covariance', {})
            )
        else:
            mean1step, cov1step = initial_state

        assert n_step > 0
        if input is None:
            inputs = []
            if out_timesteps is None:
                raise RuntimeError("If `input` is None must pass `out_timesteps`")
            num_groups = mean1step.shape[0]
        else:
            inputs = input.unbind(1)
            if out_timesteps is None:
                out_timesteps = len(inputs)
            num_groups = inputs[0].shape[0]

        # avoid 'trying to backward a second time' error:
        # TODO: move somewhere else?
        self.process_covariance.cache.clear()
        self.measure_covariance.cache.clear()

        # build design-mats:
        tv_kwargs = list(time_varying_kwargs.keys())
        Fs: List[Tensor] = []
        Hs: List[Tensor] = []
        Qs: List[Tensor] = []
        Rs: List[Tensor] = []
        for t in range(out_timesteps):
            design_kwargs = self._get_design_kwargs_for_time(t, static_kwargs, time_varying_kwargs)
            F, H, Q, R = self.get_design_mats(num_groups=num_groups, design_kwargs=design_kwargs, tv_kwargs=tv_kwargs)
            Fs += [F]
            Hs += [H]
            Qs += [Q]
            Rs += [R]

        # generate predictions:
        means: List[Tensor] = []
        covs: List[Tensor] = []
        for ts in range(out_timesteps):
            # ts: the time of the state
            # tu: the time of the update
            tu = ts - n_step
            if tu >= 0:
                mean1step, cov1step = self.kf_step.update(inputs[tu], mean1step, cov1step, H=Hs[tu], R=Rs[tu])
                mean1step, cov1step = self.kf_step.predict(mean1step, cov1step, F=Fs[tu], Q=Qs[tu])
            mean, cov = mean1step, cov1step
            for h in range(1, n_step):
                mean, cov = self.kf_step.predict(mean, cov, F=Fs[tu + h], Q=Qs[tu + h])
            means += [mean]
            covs += [cov]
        return KFOutput(torch.stack(means, 1), torch.stack(covs, 1), torch.stack(Rs, 1), torch.stack(Hs, 1))

    def _get_design_kwargs_for_time(
            self,
            time: int,
            static_kwargs: Dict[str, Dict[str, Tensor]],
            time_varying_kwargs: Dict[str, Dict[str, List[Tensor]]]
    ) -> Dict[str, Dict[str, Tensor]]:
        design_kwargs = static_kwargs.copy()
        for id_, id_tv_kwargs in time_varying_kwargs.items():
            if id_ not in design_kwargs.keys():
                design_kwargs[id_] = {}
            else:
                design_kwargs[id_] = design_kwargs[id_].copy()
            for k, v in id_tv_kwargs.items():
                design_kwargs[id_][k] = v[time]
        return design_kwargs
