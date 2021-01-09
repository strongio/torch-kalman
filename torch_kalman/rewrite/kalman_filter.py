from typing import Tuple, List, Optional, Sequence, Dict

import torch
from torch import jit, nn, Tensor

from torch_kalman.rewrite.gaussian import Gaussian
from torch_kalman.rewrite.process import Process


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
        # TODO
        return {'process_args_groupwise': {}, 'process_args_timewise': {}}

    def forward(self,
                input: Tensor,
                state: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
        return self.script_module(input=input, state=None, **self._parse_process_args())


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
        state_rank = 0
        for p in processes:
            self.processes.append(p)
            if not p.measure:
                raise RuntimeError(f"Must call `set_measure()` on '{p.id}'")
            self.process_to_slice[p.id] = (state_rank, state_rank + len(p.state_elements))
            state_rank += len(p.state_elements)
        self.state_rank = state_rank

        # XXX:
        self._Q = torch.eye(self.state_rank)
        self._R = torch.eye(len(self.measures))

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
        Q = self._Q.expand(num_groups, -1, -1)
        R = self._R.expand(num_groups, -1, -1)

        return F, H, Q, R

    @jit.script_method
    def forward(self,
                input: Tensor,
                process_args_groupwise: Dict[str, List[Tensor]],
                process_args_timewise: Dict[str, List[Tensor]],
                state: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
        if state is None:
            mean, cov = self.get_initial_state(input)
        else:
            mean, cov = state

        inputs = input.unbind(1)

        means = torch.jit.annotate(List[Tensor], [mean])
        covs = torch.jit.annotate(List[Tensor], [cov])
        for i in range(len(inputs) - 1):
            process_args = process_args_groupwise.copy()
            for k, args in process_args_timewise.items():
                # TODO: i + 1 for H?
                if k not in process_args.keys():
                    process_args[k] = []
                process_args[k].extend([v[:, i] for v in args])
            F, H, Q, R = self.get_design_mats(input=input, process_args=process_args)
            mean, cov = self.kf_step(input=inputs[i], mean=mean, cov=cov, F=F, H=H, Q=Q, R=R)
            means += [mean]
            covs += [cov]
        return torch.stack(means, 1), torch.stack(covs, 1)
