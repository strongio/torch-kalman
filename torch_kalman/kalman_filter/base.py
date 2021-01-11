from collections import namedtuple, defaultdict
from typing import Tuple, List, Optional, Sequence, Dict
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

    def _parse_process_kwargs(self, **kwargs) -> Dict[str, dict]:
        out = {'static_process_kwargs': defaultdict(dict), 'timevarying_process_kwargs': defaultdict(dict)}
        unused = set(kwargs)
        for process in self.script_module.processes:
            for key in process.expected_kwargs:
                specific_key = f"{process.id}__{key}"
                if specific_key in kwargs:
                    value = kwargs[specific_key]
                    unused.discard(specific_key)
                else:
                    value = kwargs[key]
                    unused.discard(key)
                if len(value.shape) == 3:
                    assert not getattr(value, 'requires_grad', False), f"`{specific_key}` should not require grad"
                    out['timevarying_process_kwargs'][process.id][key] = value.unbind(1)
                else:
                    out['static_process_kwargs'][process.id][key] = value
        if unused and unused != {'input'}:
            warn(f"There are unused keyword arguments:\n{unused}")
        out['static_process_kwargs'] = dict(out['static_process_kwargs'])
        out['timevarying_process_kwargs'] = dict(out['timevarying_process_kwargs'])
        return out

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
                **self._parse_process_kwargs(input=input, **kwargs)
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

        if process_covariance is None:
            process_covariance = Covariance(rank=self.state_rank, empty_idx=self.no_pcov_idx)
        self.process_covariance = process_covariance
        if measure_covariance is None:
            measure_covariance = Covariance(rank=len(self.measures))
        self.measure_covariance = measure_covariance
        if initial_covariance is None:
            initial_covariance = Covariance(rank=self.state_rank)  # TODO: fixed state-elements; low-rank
        self.initial_covariance = initial_covariance

    def get_initial_state(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        num_groups = input.shape[0]
        mean = torch.zeros(num_groups, self.state_rank)  # TODO
        cov = self.initial_covariance(input)
        return mean, cov

    def get_design_mats(self,
                        input: Tensor,
                        process_kwargs: Dict[str, Dict[str, Tensor]],
                        tv_kwargs: List[str]
                        ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        num_groups = input.shape[0]

        F = torch.zeros((num_groups, self.state_rank, self.state_rank))
        H = torch.zeros((num_groups, len(self.measures), self.state_rank))
        for process in self.processes:
            if process.id in process_kwargs.keys():
                this_process_kwargs = process_kwargs[process.id]
            else:
                this_process_kwargs = {}
            pH, pF = process(this_process_kwargs, tv_kwargs=tv_kwargs)

            _process_slice = slice(*self.process_to_slice[process.id])
            H[:, self.measure_to_idx[process.measure], _process_slice] = pH
            F[:, _process_slice, _process_slice] = pF

        Q = self.process_covariance(input)
        R = self.measure_covariance(input)

        return F, H, Q, R

    def forward(self,
                input: Tensor,
                static_process_kwargs: Dict[str, Dict[str, Tensor]],
                timevarying_process_kwargs: Dict[str, Dict[str, List[Tensor]]],
                n_step: int = 1,
                out_timesteps: Optional[int] = None,
                initial_state: Optional[Tuple[Tensor, Tensor]] = None) -> KFOutput:

        assert n_step > 0
        if input is None:
            raise NotImplementedError("TODO")
        else:
            inputs = input.unbind(1)
            if out_timesteps is None:
                out_timesteps = len(inputs)

        # avoid 'trying to backward a second time' error:
        # TODO: move somewhere else?
        self.process_covariance.cache.clear()
        self.measure_covariance.cache.clear()

        # build design-mats:
        tv_kwargs = list(timevarying_process_kwargs.keys())
        Fs: List[Tensor] = []
        Hs: List[Tensor] = []
        Qs: List[Tensor] = []
        Rs: List[Tensor] = []
        for t in range(out_timesteps):
            # get design-mats for this timestep:
            process_kwargs = static_process_kwargs.copy()
            for pid, tv_pkwargs in timevarying_process_kwargs.items():
                if pid not in process_kwargs.keys():
                    process_kwargs[pid] = {}
                else:
                    process_kwargs[pid] = process_kwargs[pid].copy()
                for k, v in tv_pkwargs.items():
                    process_kwargs[pid][k] = v[t]
            F, H, Q, R = self.get_design_mats(input=input, process_kwargs=process_kwargs, tv_kwargs=tv_kwargs)
            Fs += [F]
            Hs += [H]
            Qs += [Q]
            Rs += [R]

        # generate predictions:
        if initial_state is None:
            mean1step, cov1step = self.get_initial_state(input)
        else:
            mean1step, cov1step = initial_state
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
