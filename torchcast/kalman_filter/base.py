from collections import defaultdict
from typing import Tuple, List, Optional, Sequence, Dict, Iterable, Callable
from warnings import warn

import torch
from torch import nn, Tensor

from torchcast.covariance import Covariance
from torchcast.kalman_filter.gaussian import GaussianStep
from torchcast.kalman_filter.predictions import Predictions
from torchcast.kalman_filter.simulations import Simulations
from torchcast.process.regression import Process


class KalmanFilter(nn.Module):
    """
    The KalmanFilter is a `nn.Module` which generates predictions and forecasts using a state-space model. Processes
    are used to specify how latent-states translate into the measurable data being forecasted.

    Parameters
    ----------
    :param processes: A list of :class:`.Process` modules.
    :param measures: A list of strings specifying the names of the dimensions of the time-series being measured.
    :param process_covariance: A module created with ``Covariance.from_processes(processes, cov_type='process')``.
    :param measure_covariance: A module created with ``Covariance.from_measures(measures)``.
    :param initial_covariance: A module created with ``Covariance.from_processes(processes, cov_type='initial')``.
    :param compiled: Should the core modules be passed through :class:`torch.jit.script` to compile them to
     TorchScript? Can be disabled if compilation issues arise.
    :param kwargs: Further arguments passed to ScriptKalmanFilter's child-classes (base-class takes no kwargs).
    """
    kf_step_cls = GaussianStep

    def __init__(self,
                 processes: Sequence[Process],
                 measures: Optional[Sequence[str]] = None,
                 process_covariance: Optional[Covariance] = None,
                 measure_covariance: Optional[Covariance] = None,
                 initial_covariance: Optional[Covariance] = None):
        super(KalmanFilter, self).__init__()

        self._validate(processes, measures)

        # covariances:
        if process_covariance is None:
            process_covariance = Covariance.for_processes(processes, cov_type='process')
        self.process_covariance = process_covariance.set_id('process_covariance')

        if measure_covariance is None:
            measure_covariance = Covariance.for_measures(measures)
        self.measure_covariance = measure_covariance.set_id('measure_covariance')

        if initial_covariance is None:
            initial_covariance = Covariance.for_processes(processes, cov_type='initial')
        self.initial_covariance = initial_covariance.set_id('initial_covariance')

        self.kf_step = self.kf_step_cls()

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

        # can disable for debugging/tests:
        self._scale_by_measure_var = True

    def fit(self,
            y: Tensor,
            tol: float = .001,
            patience: int = 3,
            max_iter: int = 200,
            optimizer: Optional[torch.optim.Optimizer] = None,
            verbose: int = 2,
            callbacks: Sequence[Callable] = (),
            **kwargs):

        if optimizer is None:
            optimizer = torch.optim.LBFGS(self.parameters(), max_iter=10, line_search_fn='strong_wolfe', lr=.5)

        self.set_initial_values(y)

        prog = None
        if verbose > 1 and isinstance(optimizer, torch.optim.LBFGS):
            from tqdm.auto import tqdm
            prog = tqdm(total=optimizer.param_groups[0]['max_eval'])

        epoch = 0

        def closure():
            optimizer.zero_grad()
            pred = self(y, **kwargs)
            loss = -pred.log_prob(y).mean()
            loss.backward()
            if prog:
                prog.update()
                prog.set_description(f'Epoch: {epoch}; Loss: {loss:.4f}')
            return loss

        prev_train_loss = float('inf')
        num_lower = 0
        for epoch in range(max_iter):
            if prog:
                prog.reset()
            train_loss = optimizer.step(closure).item()
            for callback in callbacks:
                callback(train_loss)
            if abs(train_loss - prev_train_loss) < tol:
                num_lower += 1
            else:
                num_lower = 0
            if num_lower == patience:
                break

            prev_train_loss = train_loss

        return self

    @torch.jit.ignore()
    def set_initial_values(self, y: Tensor):

        assert len(self.measures) == y.shape[-1]

        hits = {m: [] for m in self.measures}
        for pid, process in self.named_processes():
            # have to use the name since `jit.script()` strips the class
            if (getattr(process, 'original_name', None) or type(process).__name__) in ('LocalLevel', 'LocalTrend'):
                if 'position->position' in (process.f_modules or {}):
                    continue
                assert process.measure

                hits[process.measure].append(pid)
                measure_idx = list(self.measures).index(process.measure)
                with torch.no_grad():
                    process.init_mean[0] = y[:, 0, measure_idx].mean()

        for measure, procs in hits.items():
            if len(procs) > 1:
                warn(
                    f"For measure '{measure}', multiple processes ({procs}) track the overall level; consider adding "
                    f"`decay` to all but one."
                )
            elif not len(procs):
                warn(
                    f"For measure '{measure}', no processes track the overall level; consider centering data in "
                    f"preprocessing prior to training (if you haven't already)."
                )

    @staticmethod
    def _validate(processes: Sequence[Process], measures: Sequence[str]):
        if isinstance(measures, str):
            measures = [measures]
            warn(f"`measures` should be a list of strings not a string; interpreted as `{measures}`.")
        elif not hasattr(measures, '__getitem__'):
            warn(f"`measures` appears to be an unordered collection")

        for p in processes:
            if isinstance(p, torch.jit.RecursiveScriptModule):
                raise TypeError(
                    f"Processes should not be wrapped in `torch.jit.script` *before* being passed to `KalmanFilter`"
                )
            if p.measure:
                if p.measure not in measures:
                    raise RuntimeError(f"'{p.id}' has measure '{p.measure}' not in `measures`.")
            else:
                if len(measures) > 1:
                    raise RuntimeError(f"Must set measure for '{p.id}' since there are multiple measures.")
                p.measure = measures[0]

    @torch.jit.ignore()
    def named_processes(self) -> Iterable[Tuple[str, Process]]:
        for pid in self.processes:
            yield pid, self.processes[pid]

    @torch.jit.ignore()
    def named_covariances(self) -> Iterable[Tuple[str, Covariance]]:
        return [
            ('process_covariance', self.process_covariance),
            ('measure_covariance', self.measure_covariance),
            ('initial_covariance', self.initial_covariance),
        ]

    @torch.jit.ignore()
    def forward(self,
                input: Optional[Tensor],
                n_step: int = 1,
                out_timesteps: Optional[int] = None,
                initial_state: Optional[Tuple[Tensor, Tensor]] = None,
                every_step: bool = True,
                **kwargs) -> Predictions:
        """
        Generate n-step-ahead predictions from the model.

        :param input: A (group X time X measures) tensor. Optional if `initial_state` is specified.
        :param n_step: What is the horizon for the predictions output for each timepoint? Defaults to one-step-ahead
         predictions (i.e. n_step=1).
        :param out_timesteps: The number of timesteps to produce in the output. This is useful when passing a tensor
         of predictors that goes later in time than the `input` tensor -- you can specify `out_timesteps=X.shape[1]` to
         get forecasts into this later time horizon.
        :param initial_state: Optional, default is `None` (with the initial state being determined internally). Can
         pass a `mean`, `cov` tuple from a previous call.
        :param every_step: By default, `n_step` ahead predictions will be generated at every timestep. If
         `every_step=False`, then these predictions will only be generated every `n_step` timesteps. For example, with
         hourly data, `n_step=24` and every_step=True, each timepoint would be a forecast generated with data 24-hours
         in the past. But with `every_step=False` in this case, then the first timestep would be 1-step-ahead, the 2nd
         would be 2-step-ahead, ... the 23rd would be 24-step-ahead, the 24th would be 1-step-ahead, etc. The advantage
         to every_step=False is speed: training data for long-range forecasts can be generated without requiring the
         model to produce and discard intermediate predictions every timestep.
        :param kwargs: Further arguments passed to the `processes`. For example, many seasonal processes require a
         `state_datetimes` argument; the `LinearModel` and `NN` processes expect a `X` argument for predictors.
        :return: A `Predictions` object with `log_prob()` and `to_dataframe()` methods.
        """

        if out_timesteps is None and input is None:
            raise RuntimeError("If `input` is None must specify `out_timesteps`")

        means, covs, R, H = self._script_forward(
            input=input,
            initial_state=initial_state,
            n_step=n_step,
            every_step=every_step,
            out_timesteps=out_timesteps,
            _disable_cache=kwargs.pop('_disable_cache', False),
            **self._parse_design_kwargs(input=input, out_timesteps=out_timesteps or input.shape[1], **kwargs)
        )
        return Predictions(state_means=means, state_covs=covs, R=R, H=H, kalman_filter=self)

    @torch.jit.export
    def _script_forward(self,
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

    @torch.no_grad()
    @torch.jit.ignore()
    def simulate(self,
                 out_timesteps: int,
                 initial_state: Optional[Tuple[Tensor, Tensor]] = None,
                 num_sims: Optional[int] = None,
                 progress: bool = False,
                 **kwargs):
        """
        Generate simulated state-trajectories from your model.

        :param out_timesteps: The number of timesteps to generate in the output.
        :param initial_state: The initial state of the system: a tuple of `mean`, `cov`.
        :param num_sims: The number of state-trajectories to simulate.
        :param progress: Should a progress-bar be displayed? Requires `tqdm`.
        :param kwargs: Further arguments passed to the `processes`.
        :return: A `Simulations` object with a `sample()` method.
        """

        design_kwargs = self._parse_design_kwargs(input=None, out_timesteps=out_timesteps, **kwargs)
        design_cache = {}

        if initial_state is None:
            init_mean_kwargs = design_kwargs.pop('init_mean_kwargs')
            init_cov_kwargs = design_kwargs['static_kwargs'].pop('initial_covariance', {})
            if num_sims is None:
                raise RuntimeError("Must pass `initial_state` or `num_sims`")
            design_kwargs_t = self._get_design_kwargs_for_time(0, **design_kwargs)
            *_, R = self.get_design_mats(
                num_groups=num_sims, design_kwargs=design_kwargs_t, cache=design_cache
            )
            mean, cov = self.get_initial_state(
                input=torch.zeros((num_sims, len(self.measures))),
                init_mean_kwargs=init_mean_kwargs,
                init_cov_kwargs=init_cov_kwargs,
                measure_cov=R
            )
        else:
            if num_sims is not None:
                raise RuntimeError("Cannot pass both `num_sims` and `initial_state`")
            mean, cov = initial_state

        times = range(out_timesteps)
        if progress:
            if progress is True:
                try:
                    from tqdm.auto import tqdm
                    progress = tqdm
                except ImportError:
                    warn("`progress=True` requires package `tqdm`.")
                    progress = lambda x: x
            times = progress(times)

        dist_cls = self.kf_step.get_distribution()

        means: List[Tensor] = []
        Hs: List[Tensor] = []
        Rs: List[Tensor] = []
        for t in times:
            mean = dist_cls(mean, cov).rsample()
            design_kwargs_t = self._get_design_kwargs_for_time(t, **design_kwargs)
            F, H, Q, R = self.get_design_mats(
                num_groups=num_sims, design_kwargs=design_kwargs_t, cache=design_cache
            )
            mean, cov = self.kf_step.predict(mean, .0001 * torch.eye(mean.shape[-1]), F=F, Q=Q)
            means += [mean]
            Rs += [R]
            Hs += [H]

        return Simulations(torch.stack(means, 1), H=torch.stack(Hs, 1), R=torch.stack(Rs, 1), kalman_filter=self)

    @torch.jit.ignore()
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
                    pf = process.f_forward(design_kwargs.get(pid, {}), cache=cache)
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
                    ph = process.h_forward(design_kwargs.get(pid, {}), cache=cache)
                    if torch.isnan(ph).any():
                        raise RuntimeError(f"{process.id} produced H with nans")
                    cache['base_H'][:, self.measure_to_idx[process.measure], _process_slice] = ph

        H = cache['base_H'].clone()
        F = cache['base_F'].clone()
        for pid, process in self.processes.items():
            if process.time_varying_kwargs is not None:
                _process_slice = slice(*self.process_to_slice[pid])
                if process.h_kwarg in process.time_varying_kwargs:
                    ph = process.h_forward(design_kwargs.get(pid, {}), cache=cache)
                    if torch.isnan(ph).any():
                        raise RuntimeError(f"{process.id} produced H with nans")
                    H[:, self.measure_to_idx[process.measure], _process_slice] = ph

                if process.f_kwarg in process.time_varying_kwargs:
                    pf = process.f_forward(design_kwargs.get(pid, {}), cache=cache)
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

    def __repr__(self) -> str:
        return f'{type(self).__name__}' \
               f'(processes={repr(list(self.processes.values()))}, measures={repr(list(self.measures))})'
