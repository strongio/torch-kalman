from collections import namedtuple
from typing import Tuple, List, Optional, Sequence, Dict

import torch
from torch import nn, Tensor

from torch_kalman.covariance import Covariance
from torch_kalman.kalman_filter.gaussian import GaussianStep
from torch_kalman.kalman_filter.state_belief_over_time import StateBeliefOverTime
from torch_kalman.process.regression import Process

KFOutput = namedtuple('KFOutput', ['means', 'covs', 'R', 'H'])


class KalmanFilter(nn.Module):
    kf_step = GaussianStep
    use_jit = True

    def __init__(self, processes: Sequence[Process], measures: Sequence[str]):
        super(KalmanFilter, self).__init__()
        self._validate(processes, measures)
        self.script_module = ScriptKalmanFilter(kf_step=self.kf_step(), processes=processes, measures=measures)
        if self.use_jit:
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

    def _parse_process_args(self, *args, **kwargs) -> Dict[str, dict]:
        out = {'process_kwargs_groupwise': {}, 'process_kwargs_timewise': {}}
        # TODO: track used/unused?
        for process in self.script_module.processes:
            out['process_kwargs_groupwise'][process.id] = process.get_groupwise_kwargs(*args, **kwargs)
            out['process_kwargs_timewise'][process.id] = process.get_timewise_kwargs(*args, **kwargs)
        return out

    def forward(self,
                input: Tensor,
                n_step: int = 1,
                out_timesteps: Optional[int] = None,
                initial_state: Optional[Tuple[Tensor, Tensor]] = None,
                **kwargs) -> StateBeliefOverTime:
        means, covs, R, H = self.script_module(
            input=input,
            initial_state=initial_state,
            n_step=n_step,
            out_timesteps=out_timesteps,
            **self._parse_process_args(input=input, **kwargs)
        )
        return StateBeliefOverTime(means, covs, R=R, H=H, kf_step=self.kf_step)


class ScriptKalmanFilter(nn.Module):
    def __init__(self,
                 kf_step: 'GaussianStep',
                 processes: Sequence[Process],
                 measures: Sequence[str]):
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

        self.process_covariance = Covariance(rank=self.state_rank, empty_idx=self.no_pcov_idx)
        self.measure_covariance = Covariance(rank=len(self.measures))

    def get_initial_state(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        # TODO
        num_groups = input.shape[0]
        mean = torch.zeros(num_groups, self.state_rank)
        cov = torch.eye(self.state_rank).expand(num_groups, -1, -1)
        return mean, cov

    def get_design_mats(self,
                        input: Tensor,
                        process_kwargs: Dict[str, Dict[str, Tensor]]
                        ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        num_groups = input.shape[0]

        F = torch.zeros((num_groups, self.state_rank, self.state_rank))
        H = torch.zeros((num_groups, len(self.measures), self.state_rank))
        for process in self.processes:
            if process.id in process_kwargs.keys():
                this_process_kwargs = process_kwargs[process.id]
            else:
                this_process_kwargs = {}
            pH, pF = process(this_process_kwargs)

            _process_slice = slice(*self.process_to_slice[process.id])
            H[:, self.measure_to_idx[process.measure], _process_slice] = pH
            F[:, _process_slice, _process_slice] = pF

        Q = self.process_covariance(input)
        R = self.measure_covariance(input)

        return F, H, Q, R

    def forward(self,
                input: Tensor,
                process_kwargs_groupwise: Dict[str, Dict[str, Tensor]],
                process_kwargs_timewise: Dict[str, Dict[str, Tensor]],
                n_step: int = 1,
                out_timesteps: Optional[int] = None,
                initial_state: Optional[Tuple[Tensor, Tensor]] = None) -> KFOutput:
        if initial_state is None:
            mean, cov = self.get_initial_state(input)
        else:
            mean, cov = initial_state

        assert n_step > 0
        if input is None:
            raise NotImplementedError("TODO")
        else:
            inputs = input.unbind(1)
            if out_timesteps is None:
                out_timesteps = len(inputs)

        # avoid 'trying to backward a second time' error:
        for p in self.processes:
            p.cache.clear()
        self.process_covariance.cache.clear()
        self.measure_covariance.cache.clear()

        means: List[Tensor] = []
        covs: List[Tensor] = []
        Hs: List[Tensor] = []
        Rs: List[Tensor] = []
        for t in range(out_timesteps):
            # get design-mats for this timestep:
            process_kwargs = process_kwargs_groupwise.copy()
            for pid, pkwargs in process_kwargs_timewise.items():
                if pid not in process_kwargs.keys():
                    process_kwargs[pid] = {}
                for k, v in pkwargs.items():
                    process_kwargs[pid][k] = v
            F, H, Q, R = self.get_design_mats(input=input, process_kwargs=process_kwargs)

            # get new mean/cov
            if n_step <= t <= len(inputs):
                # don't update until we have input
                mean, cov = self.kf_step.update(input=inputs[t - 1], mean=mean, cov=cov, H=H, R=R)
                if t >= n_step:
                    # if t < n_step, then it doesn't make sense to increase uncertainty with each step -- our initial
                    # state already represents maximum uncertainty
                    mean, cov = self.kf_step.predict(mean, cov, F=F, Q=Q)
                    assert n_step == 1  # TODO

            means += [mean]
            covs += [cov]
            Hs += [H]
            Rs += [R]
        return KFOutput(torch.stack(means, 1), torch.stack(covs, 1), torch.stack(Rs, 1), torch.stack(Hs, 1))
