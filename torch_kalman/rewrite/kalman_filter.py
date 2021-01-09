from collections import namedtuple
from typing import Tuple, List, Optional, Sequence, Dict

import torch
from torch import jit, nn, Tensor

from torch_kalman.rewrite.covariance import Covariance
from torch_kalman.rewrite.gaussian import Gaussian
from torch_kalman.rewrite.process import Process

KFOutput = namedtuple('KFOutput', ['means', 'covs', 'R', 'H'])


class KalmanFilter(nn.Module):
    kf_step = Gaussian

    def __init__(self, processes: Sequence[Process], measures: Sequence[str]):
        super(KalmanFilter, self).__init__()
        self._validate(processes, measures)
        self.script_module = ScriptKalmanFilter(kf_step=self.kf_step(), processes=processes, measures=measures)

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
        out = {'process_args_groupwise': {}, 'process_args_timewise': {}}
        for process in self.script_module.processes:
            out['process_args_groupwise'][process.id] = process.get_groupwise_args(*args, **kwargs)
            out['process_args_timewise'][process.id] = process.get_timewise_args(*args, **kwargs)
        return out

    def forward(self,
                input: Tensor,
                n_step: int = 1,
                out_timesteps: Optional[int] = None,
                initial_state: Optional[Tuple[Tensor, Tensor]] = None,
                **kwargs) -> KFOutput:
        return self.script_module(
            input=input,
            initial_state=initial_state,
            n_step=n_step,
            out_timesteps=out_timesteps,
            **self._parse_process_args(input=input, **kwargs)
        )


class ScriptKalmanFilter(jit.ScriptModule):
    def __init__(self,
                 kf_step: 'Gaussian',
                 processes: Sequence[Process],
                 measures: Sequence[str]):
        super(ScriptKalmanFilter, self).__init__()

        self.kf_step = kf_step

        # measures:
        self.measures = measures
        self.measure_to_idx = {m: i for i, m in enumerate(self.measures)}

        # processes:
        self.processes = nn.ModuleList()
        self.process_to_slice = torch.jit.annotate(Dict[str, Tuple[int, int]], {})
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
                        process_args: Dict[str, List[Tensor]]
                        ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        num_groups = input.shape[0]

        F = torch.zeros((num_groups, self.state_rank, self.state_rank))
        H = torch.zeros((num_groups, len(self.measures), self.state_rank))
        for process in self.processes:
            if process.id in process_args.keys():
                this_process_args = process_args[process.id]
            else:
                this_process_args = []
            pH, pF = process(this_process_args)

            _process_slice = slice(*self.process_to_slice[process.id])
            H[:, self.measure_to_idx[process.measure], _process_slice] = pH
            F[:, _process_slice, _process_slice] = pF

        if 'process_covariance' in process_args.keys():
            pcov_args = process_args['process_covariance']
        else:
            pcov_args = [input]
        Q = self.process_covariance(pcov_args)

        # TODO: scale Q by R?
        if 'measure_covariance' in process_args.keys():
            mcov_args = process_args['measure_covariance']
        else:
            mcov_args = [input]
        R = self.measure_covariance(mcov_args)

        return F, H, Q, R

    @jit.script_method
    def forward(self,
                input: Tensor,
                process_args_groupwise: Dict[str, List[Tensor]],
                process_args_timewise: Dict[str, List[Tensor]],
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

        means = torch.jit.annotate(List[Tensor], [])
        covs = torch.jit.annotate(List[Tensor], [])
        Hs = torch.jit.annotate(List[Tensor], [])
        Rs = torch.jit.annotate(List[Tensor], [])
        for t in range(out_timesteps):
            # get design-mats for this timestep:
            process_args = process_args_groupwise.copy()
            for k, args in process_args_timewise.items():
                if k not in process_args.keys():
                    process_args[k] = []
                process_args[k].extend([v[:, t] for v in args])
            F, H, Q, R = self.get_design_mats(input=input, process_args=process_args)

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
