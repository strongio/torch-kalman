from collections import defaultdict
from typing import Tuple, List, Optional, Sequence, Dict, Iterable
from warnings import warn

import torch
from torch import nn, Tensor

from torch_kalman.covariance import Covariance
from torch_kalman.kalman_filter.gaussian import GaussianStep
from torch_kalman.kalman_filter.predictions import Predictions
from torch_kalman.kalman_filter.simulations import Simulations
from torch_kalman.process.regression import Process


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
            for pid, p in self.script_module.processes.items():
                self.script_module.processes[pid] = torch.jit.script(p)
            self.script_module = torch.jit.script(self.script_module)

    @property
    def measures(self) -> Sequence[str]:
        return self.script_module.measures

    @property
    def processes(self) -> Sequence[Process]:
        return self.script_module.processes

    @staticmethod
    def _validate(processes: Sequence[Process], measures: Sequence[str]):
        assert not isinstance(measures, str)
        for p in processes:
            if isinstance(p, torch.jit.RecursiveScriptModule):
                raise TypeError("Processes should not be wrapped in `torch.jit.script`, this will be done internally.")
            if p.measure:
                if p.measure not in measures:
                    raise RuntimeError(f"'{p.id}' has measure '{p.measure}' not in `measures`.")
            else:
                if len(measures) > 1:
                    raise RuntimeError(f"Must set measure for '{p.id}' since there are multiple measures.")
                p.measure = measures[0]

    def named_processes(self) -> Iterable[Tuple[str, Process]]:
        for pid in self.script_module.processes:
            yield pid, self.script_module.processes[pid]

    def named_covariances(self) -> Iterable[Tuple[str, Covariance]]:
        return [
            ('process_covariance', self.script_module.process_covariance),
            ('measure_covariance', self.script_module.measure_covariance),
            ('initial_covariance', self.script_module.initial_covariance),
        ]

    def _parse_design_kwargs(self, input: Optional[Tensor], out_timesteps: int, **kwargs) -> Dict[str, dict]:
        static_kwargs = defaultdict(dict)
        time_varying_kwargs = defaultdict(dict)
        init_mean_kwargs = defaultdict(dict)
        unused = set(kwargs)
        kwargs.update(input=input, current_timestep=torch.tensor(list(range(out_timesteps))).view(1, -1, 1))
        for submodule_nm, submodule in list(self.named_processes()) + list(self.named_covariances()):
            for found_key, key_name, key_type, value in submodule.get_kwargs(kwargs):
                unused.discard(found_key)
                if key_type == 'init_mean':
                    init_mean_kwargs[submodule_nm][key_name] = value
                elif key_type == 'time_varying':
                    time_varying_kwargs[submodule_nm][key_name] = value.unbind(1)
                elif key_type == 'static':
                    static_kwargs[submodule_nm][key_name] = value
                else:
                    raise RuntimeError(
                        f"'{submodule_nm}' gave unknown key_type {key_type}; expected 'init_mean', 'time_varying', "
                        f"or 'static'"
                    )

        if unused:
            warn(f"There are unused keyword arguments:\n{unused}")
        return {
            'static_kwargs': dict(static_kwargs),
            'time_varying_kwargs': dict(time_varying_kwargs),
            'init_mean_kwargs': dict(init_mean_kwargs)
        }

    def forward(self,
                input: Optional[Tensor],
                n_step: int = 1,
                out_timesteps: Optional[int] = None,
                initial_state: Optional[Tuple[Tensor, Tensor]] = None,
                _disable_cache: bool = False,
                **kwargs) -> Predictions:

        if out_timesteps is None and input is None:
            raise RuntimeError("If `input` is None must specify `out_timesteps`")

        means, covs, R, H = self.script_module(
            input=input,
            initial_state=initial_state,
            n_step=n_step,
            out_timesteps=out_timesteps,
            _disable_cache=_disable_cache,
            **self._parse_design_kwargs(input=input, out_timesteps=out_timesteps or input.shape[1], **kwargs)
        )
        return Predictions(state_means=means, state_covs=covs, R=R, H=H, kalman_filter=self)

    def simulate(self,
                 out_timesteps: int,
                 initial_state: Optional[Tuple[Tensor, Tensor]] = None,
                 num_sims: Optional[int] = None,
                 progress: bool = False,
                 **kwargs):

        design_kwargs = self._parse_design_kwargs(input=None, out_timesteps=out_timesteps, **kwargs)
        design_cache = {}

        with torch.no_grad():
            if initial_state is None:
                init_mean_kwargs = design_kwargs.pop('init_mean_kwargs')
                init_cov_kwargs = design_kwargs['static_kwargs'].pop('initial_covariance', {})
                if num_sims is None:
                    raise RuntimeError("Must pass `initial_state` or `num_sims`")
                design_kwargs_t = self.script_module._get_design_kwargs_for_time(0, **design_kwargs)
                *_, R = self.script_module.get_design_mats(
                    num_groups=num_sims, design_kwargs=design_kwargs_t, cache=design_cache
                )
                mean, cov = self.script_module.get_initial_state(
                    input=torch.zeros((num_sims, len(self.measures))),
                    init_mean_kwargs=init_mean_kwargs,
                    init_cov_kwargs=init_cov_kwargs,
                    measure_cov=R
                )
            else:
                if num_sims is not None:
                    raise RuntimeError("Cannot pass both `num_sims` and `initial_state`")
                mean, cov = initial_state

            kf_step = self.kf_step()

            times = range(out_timesteps)
            if progress:
                if progress is True:
                    from tqdm.auto import tqdm
                    progress = tqdm
                times = progress(times)

            means: List[Tensor] = []
            Hs: List[Tensor] = []
            Rs: List[Tensor] = []
            for t in times:
                mean = kf_step.distribution_cls(mean, cov).rsample()
                design_kwargs_t = self.script_module._get_design_kwargs_for_time(t, **design_kwargs)
                F, H, Q, R = self.script_module.get_design_mats(
                    num_groups=num_sims, design_kwargs=design_kwargs_t, cache=design_cache
                )
                mean, cov = kf_step.predict(mean, .0001 * torch.eye(mean.shape[-1]), F=F, Q=Q)
                means += [mean]
                Rs += [R]
                Hs += [H]

        return Simulations(torch.stack(means, 1), H=torch.stack(Hs, 1), R=torch.stack(Rs, 1), kalman_filter=self)


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
        self.processes = nn.ModuleDict()
        self.process_to_slice: Dict[str, Tuple[int, int]] = {}
        self.state_rank = 0
        self.no_pcov_idx = []
        self.no_icov_idx = []
        for p in processes:
            assert p.measure, f"{p.id} does not have its `measure` set"
            self.processes[p.id] = p
            self.process_to_slice[p.id] = (self.state_rank, self.state_rank + len(p.state_elements))

            for i, se in enumerate(p.state_elements):
                if p.no_pcov_state_elements is not None and se in p.no_pcov_state_elements:
                    self.no_pcov_idx.append(self.state_rank + i)
                if p.no_icov_state_elements is not None and se in p.no_icov_state_elements:
                    self.no_icov_idx.append(self.state_rank + i)
            self.state_rank += len(p.state_elements)

        # covariances:
        if process_covariance is None:
            process_covariance = Covariance(rank=self.state_rank, empty_idx=self.no_pcov_idx)
        self.process_covariance = process_covariance.set_id('process_covariance')
        if measure_covariance is None:
            measure_covariance = Covariance(rank=len(self.measures))
        self.measure_covariance = measure_covariance.set_id('measure_covariance')
        if initial_covariance is None:
            initial_covariance = Covariance(rank=self.state_rank, empty_idx=self.no_icov_idx, method='low_rank')
        self.initial_covariance = initial_covariance.set_id('initial_covariance')
        # can disable for debugging/tests:
        self._scale_by_measure_var = True

    def get_initial_state(self,
                          input: Tensor,
                          init_mean_kwargs: Dict[str, Dict[str, Tensor]],
                          init_cov_kwargs: Dict[str, Tensor],
                          measure_cov: Tensor) -> Tuple[Tensor, Tensor]:
        num_groups = input.shape[0]

        measure_scaling = self._get_measure_scaling(measure_cov)

        # initial state mean:
        mean = torch.zeros(num_groups, self.state_rank)
        for pid, p in self.processes.items():
            _process_slice = slice(*self.process_to_slice[pid])
            mean[:, _process_slice] = p.get_initial_state_mean(init_mean_kwargs.get(pid, {}))

        mean = mean * measure_scaling

        # initial cov:
        cov = self.initial_covariance(init_cov_kwargs, cache={})
        if len(cov.shape) == 2:
            cov = cov.expand(num_groups, -1, -1)
        diag_multi = torch.diag_embed(measure_scaling)
        cov = diag_multi @ cov @ diag_multi
        return mean, cov

    def get_design_mats(self,
                        num_groups: int,
                        design_kwargs: Dict[str, Dict[str, Tensor]],
                        cache: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if cache is None:
            cache = {}
        _empty = ['_']
        if 'base_F' not in cache:
            cache['base_F'] = torch.zeros((num_groups, self.state_rank, self.state_rank))
            for pid, process in self.processes.items():
                tv_kwargs = _empty
                if process.time_varying_kwargs is not None:
                    tv_kwargs = process.time_varying_kwargs
                if process.f_kwarg not in tv_kwargs:
                    _process_slice = slice(*self.process_to_slice[pid])
                    cache['base_F'][:, _process_slice, _process_slice] = \
                        process(design_kwargs.get(pid, {}), which='f', cache=cache)
        if 'base_H' not in cache:
            cache['base_H'] = torch.zeros((num_groups, len(self.measures), self.state_rank))
            for pid, process in self.processes.items():
                tv_kwargs = _empty
                if process.time_varying_kwargs is not None:
                    tv_kwargs = process.time_varying_kwargs
                if process.h_kwarg not in tv_kwargs:
                    _process_slice = slice(*self.process_to_slice[pid])
                    cache['base_H'][:, self.measure_to_idx[process.measure], _process_slice] = \
                        process(design_kwargs.get(pid, {}), which='h', cache=cache)

        H = cache['base_H'].clone()
        F = cache['base_F'].clone()
        for pid, process in self.processes.items():
            if process.time_varying_kwargs is not None:
                _process_slice = slice(*self.process_to_slice[pid])
                if process.h_kwarg in process.time_varying_kwargs:
                    H[:, self.measure_to_idx[process.measure], _process_slice] = \
                        process(design_kwargs.get(pid, {}), which='h', cache=cache)
                if process.f_kwarg in process.time_varying_kwargs:
                    F[:, _process_slice, _process_slice] = \
                        process(design_kwargs.get(pid, {}), which='f', cache=cache)

        R = self.measure_covariance(design_kwargs.get('measure_covariance', {}), cache=cache)
        if len(R.shape) == 2:
            R = R.expand(num_groups, -1, -1)

        Q = self.process_covariance(design_kwargs.get('process_covariance', {}), cache=cache)
        if len(Q.shape) == 2:
            Q = Q.expand(num_groups, -1, -1)
        diag_multi = torch.diag_embed(self._get_measure_scaling(R))
        Q = diag_multi @ Q @ diag_multi

        return F, H, Q, R

    def _get_measure_scaling(self, measure_cov: Tensor) -> Tensor:
        Rdiag = measure_cov.diagonal(dim1=-2, dim2=-1)
        if self._scale_by_measure_var:
            multi = torch.zeros(measure_cov.shape[0:-2] + (self.state_rank,))
            for pid, process in self.processes.items():
                pidx = self.process_to_slice[pid]
                multi[:, slice(*pidx)] = Rdiag[:, self.measure_to_idx[process.measure]].sqrt().unsqueeze(-1)
            assert (multi > 0).all()
        else:
            multi = torch.ones(measure_cov.shape[0:-2] + (self.state_rank,))
        return multi

    def forward(self,
                input: Optional[Tensor],
                static_kwargs: Dict[str, Dict[str, Tensor]],
                time_varying_kwargs: Dict[str, Dict[str, List[Tensor]]],
                init_mean_kwargs: Dict[str, Dict[str, Tensor]],
                n_step: int = 1,
                out_timesteps: Optional[int] = None,
                initial_state: Optional[Tuple[Tensor, Tensor]] = None,
                _disable_cache: bool = False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

        assert n_step > 0

        init_cov_kwargs = static_kwargs.pop('initial_covariance', {})
        design_cache = {}

        if input is None:
            if out_timesteps is None:
                raise RuntimeError("If `input` is None must pass `out_timesteps`")
            if initial_state is None:
                raise RuntimeError("If `input` is None must pass `initial_state`")
            inputs = []
        else:
            if len(input.shape) != 3:
                raise ValueError(f"Expected len(input.shape) == 3 (group,time,measure)")
            if input.shape[-1] != len(self.measures):
                raise ValueError(f"Expected input.shape[-1] == {len(self.measures)} (len(self.measures))")

            inputs = input.unbind(1)
            if out_timesteps is None:
                out_timesteps = len(inputs)

            if initial_state is None:
                design_kwargs = self._get_design_kwargs_for_time(0, static_kwargs, time_varying_kwargs)
                *_, R = self.get_design_mats(
                    num_groups=inputs[0].shape[0], design_kwargs=design_kwargs, cache=design_cache
                )
                initial_state = self.get_initial_state(
                    input=input,
                    init_mean_kwargs=init_mean_kwargs,
                    init_cov_kwargs=init_cov_kwargs,
                    measure_cov=R
                )

        mean1step, cov1step = initial_state
        num_groups = mean1step.shape[0]

        # build design-mats:
        Fs: List[Tensor] = []
        Hs: List[Tensor] = []
        Qs: List[Tensor] = []
        Rs: List[Tensor] = []
        for t in range(out_timesteps):
            design_kwargs = self._get_design_kwargs_for_time(t, static_kwargs, time_varying_kwargs)
            if _disable_cache:
                design_cache = {}
            F, H, Q, R = self.get_design_mats(num_groups=num_groups, design_kwargs=design_kwargs, cache=design_cache)
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
                if tu < len(inputs):  # TODO: add unit-test
                    mean1step, cov1step = self.kf_step.update(inputs[tu], mean1step, cov1step, H=Hs[tu], R=Rs[tu])
                mean1step, cov1step = self.kf_step.predict(mean1step, cov1step, F=Fs[tu], Q=Qs[tu])
            mean, cov = mean1step, cov1step
            for h in range(1, n_step):
                mean, cov = self.kf_step.predict(mean, cov, F=Fs[tu + h], Q=Qs[tu + h])
            means += [mean]
            covs += [cov]

        return torch.stack(means, 1), torch.stack(covs, 1), torch.stack(Rs, 1), torch.stack(Hs, 1)

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
