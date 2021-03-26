from typing import Tuple, List, Optional, Sequence, Dict

import torch
from torch import nn, Tensor

from torchcast.covariance import Covariance
from torchcast.kalman_filter.gaussian import GaussianStep
from torchcast.process.regression import Process


class ScriptKalmanFilter(nn.Module):

    def __init__(self,
                 kf_step: 'GaussianStep',
                 processes: Sequence[Process],
                 measures: Sequence[str],
                 process_covariance: Covariance,
                 measure_covariance: Covariance,
                 initial_covariance: Covariance):
        super(ScriptKalmanFilter, self).__init__()

        self.kf_step = kf_step

        # measures:
        self.measures = measures
        self.measure_to_idx = {m: i for i, m in enumerate(self.measures)}

        # processes:
        self.processes = nn.ModuleDict()
        self.process_to_slice: Dict[str, Tuple[int, int]] = {}
        self.state_rank = 0
        for p in processes:
            assert p.measure, f"{p.id} does not have its `measure` set"
            self.processes[p.id] = p
            self.process_to_slice[p.id] = (self.state_rank, self.state_rank + len(p.state_elements))
            self.state_rank += len(p.state_elements)

        self.process_covariance = process_covariance.set_id('process_covariance')
        self.measure_covariance = measure_covariance.set_id('measure_covariance')
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
        mean = torch.zeros(num_groups, self.state_rank, dtype=measure_scaling.dtype, device=measure_scaling.device)
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

        # measure-variance:
        R = self.measure_covariance(design_kwargs.get('measure_covariance', {}), cache=cache)
        if torch.isnan(R).any():
            raise RuntimeError(
                "`nans` in measurement covariance; possible caused by nan-gradients in training. Check your inputs."
            )
        if len(R.shape) == 2:
            R = R.expand(num_groups, -1, -1)

        # process-variance:
        Q_raw = self.process_covariance(design_kwargs.get('process_covariance', {}), cache=cache)
        if len(Q_raw.shape) == 2:
            Q_raw = Q_raw.expand(num_groups, -1, -1)

        # cache the scaling operation, which is not cheap:
        if self.process_covariance.time_varying_kwargs is None and self.measure_covariance.time_varying_kwargs is None:
            if 'q_scaled' not in cache.keys():
                measure_scaling = torch.diag_embed(self._get_measure_scaling(R))
                cache['q_scaled'] = measure_scaling @ Q_raw @ measure_scaling
            Q = cache['q_scaled']
        else:
            measure_scaling = torch.diag_embed(self._get_measure_scaling(R))
            Q = measure_scaling @ Q_raw @ measure_scaling

        _empty = ['_']
        if 'base_F' not in cache:
            cache['base_F'] = \
                torch.zeros((num_groups, self.state_rank, self.state_rank), dtype=R.dtype, device=R.device)
            for pid, process in self.processes.items():
                tv_kwargs = _empty
                if process.time_varying_kwargs is not None:
                    tv_kwargs = process.time_varying_kwargs
                if process.f_kwarg not in tv_kwargs:
                    _process_slice = slice(*self.process_to_slice[pid])
                    pf = process(design_kwargs.get(pid, {}), which='f', cache=cache)
                    if torch.isnan(pf).any():
                        raise RuntimeError(f"{process.id} produced F with nans")
                    cache['base_F'][:, _process_slice, _process_slice] = pf

        if 'base_H' not in cache:
            cache['base_H'] = \
                torch.zeros((num_groups, len(self.measures), self.state_rank), dtype=R.dtype, device=R.device)
            for pid, process in self.processes.items():
                tv_kwargs = _empty
                if process.time_varying_kwargs is not None:
                    tv_kwargs = process.time_varying_kwargs
                if process.h_kwarg not in tv_kwargs:
                    _process_slice = slice(*self.process_to_slice[pid])
                    ph = process(design_kwargs.get(pid, {}), which='h', cache=cache)
                    if torch.isnan(ph).any():
                        raise RuntimeError(f"{process.id} produced H with nans")
                    cache['base_H'][:, self.measure_to_idx[process.measure], _process_slice] = ph

        H = cache['base_H'].clone()
        F = cache['base_F'].clone()
        for pid, process in self.processes.items():
            if process.time_varying_kwargs is not None:
                _process_slice = slice(*self.process_to_slice[pid])
                if process.h_kwarg in process.time_varying_kwargs:
                    ph = process(design_kwargs.get(pid, {}), which='h', cache=cache)
                    if torch.isnan(ph).any():
                        raise RuntimeError(f"{process.id} produced H with nans")
                    H[:, self.measure_to_idx[process.measure], _process_slice] = ph

                if process.f_kwarg in process.time_varying_kwargs:
                    pf = process(design_kwargs.get(pid, {}), which='f', cache=cache)
                    if torch.isnan(pf).any():
                        raise RuntimeError(f"{process.id} produced F with nans")
                    F[:, _process_slice, _process_slice] = pf

        return F, H, Q, R

    def _get_measure_scaling(self, measure_cov: Tensor) -> Tensor:
        if self._scale_by_measure_var:
            measure_var = measure_cov.diagonal(dim1=-2, dim2=-1)
            multi = torch.zeros(measure_cov.shape[0:-2] + (self.state_rank,),
                                dtype=measure_cov.dtype, device=measure_cov.device)
            for pid, process in self.processes.items():
                pidx = self.process_to_slice[pid]
                multi[..., slice(*pidx)] = measure_var[..., self.measure_to_idx[process.measure]].sqrt().unsqueeze(-1)
            assert (multi > 0).all()
        else:
            multi = torch.ones(
                measure_cov.shape[0:-2] + (self.state_rank,), dtype=measure_cov.dtype, device=measure_cov.device
            )
        return multi

    def forward(self,
                input: Optional[Tensor],
                static_kwargs: Dict[str, Dict[str, Tensor]],
                time_varying_kwargs: Dict[str, Dict[str, List[Tensor]]],
                init_mean_kwargs: Dict[str, Dict[str, Tensor]],
                n_step: int = 1,
                out_timesteps: Optional[int] = None,
                initial_state: Optional[Tuple[Tensor, Tensor]] = None,
                every_step: bool = True,
                _disable_cache: bool = False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        :param input: A (group X time X measures) tensor. Optional if `initial_state` is specified.
        :param static_kwargs: Keyword-arguments to the Processes which do not vary over time.
        :param time_varying_kwargs: Keyword-arguments to the Process which do vary over time. At each timestep, each
        kwarg gets sliced for that timestep.
        :param init_mean_kwargs: Keyword-arguments passed to `get_initial_state`
        :param n_step: What is the horizon for predictions? Defaults to one-step-ahead (i.e. n_step=1).
        :param out_timesteps: The number of timesteps in the output. Might be longer than input if forecasting.
        :param initial_state: A (mean, cov) tuple to use at the initial state; otherwise `get_initial_state` is called.
        :param every_step: Experimental. When n_step>1, we can generate these n-step-ahead predictions at every
        timestep (e.g. 24-hour-ahead predictions every hour), in which case we'd save the 24-step-ahead prediction.
        Alternatively, we could generate 24-hour-ahead predictions at every 24th hour, in which case we'd save
        predictions 1-24. The former corresponds to every_step=True, the latter to every_step=False. If n_step=1
        (the default) then this option has no effect.
        :return: means, covs, R, H
        """
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
                if tu < len(inputs):
                    mean1step, cov1step = self.kf_step.update(inputs[tu], mean1step, cov1step, H=Hs[tu], R=Rs[tu])
                mean1step, cov1step = self.kf_step.predict(mean1step, cov1step, F=Fs[tu], Q=Qs[tu])
            # - if n_step=1, append to output immediately, and exit the loop
            # - if n_step>1 & every_step, wait to append to output until h reaches n_step
            # - if n_step>1 & !every_step, only append every 24th iter; but when we do, append for each h
            if every_step or (tu % n_step) == 0:
                mean, cov = mean1step, cov1step
                for h in range(n_step):
                    if h > 0:
                        mean, cov = self.kf_step.predict(mean, cov, F=Fs[tu + h], Q=Qs[tu + h])
                    if not every_step or h == (n_step - 1):
                        means += [mean]
                        covs += [cov]

        means = means[:out_timesteps]
        covs = covs[:out_timesteps]

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
