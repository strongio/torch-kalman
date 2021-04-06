from collections import defaultdict
from typing import Tuple, List, Optional, Sequence, Dict, Iterable, Callable, Union
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

        # the initial mean
        self.initial_mean = torch.nn.Parameter(.1 * torch.randn(self.state_rank))

        # can disable for debugging/tests:
        self._scale_by_measure_var = True

    @torch.jit.ignore()
    def fit(self,
            y: Tensor,
            tol: float = .001,
            patience: int = 3,
            max_iter: int = 200,
            optimizer: Optional[torch.optim.Optimizer] = None,
            verbose: int = 2,
            callbacks: Sequence[Callable] = (),
            **kwargs):
        """
        A high-level interface for invoking the standard model-training boilerplate. This is helpful to common cases in
        which the number of parameters is moderate and the data fit in memory. For other cases you are encouraged to
        roll your own training loop.

        :param y: Tensor containing the batch of time-series(es), see :func:`KalmanFilter.forward()`.
        :param tol: Stopping tolerance.
        :param patience: Patience: if loss changes by less than `tol` for this many epochs, then training will be
         stopped.
        :param max_iter: The maximum number of iterations after which training will stop regardless of loss.
        :param optimizer: The optimizer to use. Default is to create an instance of :class:`torch.optim.LBFGS` with
         args ``(max_iter=10, line_search_fn='strong_wolfe', lr=.5)``.
        :param verbose: If True (default) will print the loss and epoch; for :class:`torch.optim.LBFGS` optimizer
         (the default) this progress bar will tick within each epoch to track the calls to forward.
        :param callbacks: A list of functions that will be called at the end of each epoch, which take the current
         epoch's loss value.
        :param kwargs: Further keyword-arguments passed to :func:`KalmanFilter.forward()`.
        :return: This `KalmanFilter` instance.
        """

        if optimizer is None:
            optimizer = torch.optim.LBFGS(self.parameters(), max_iter=10, line_search_fn='strong_wolfe', lr=.5)

        self.set_initial_values(y)

        prog = None
        if verbose > 1:
            from tqdm.auto import tqdm
            if isinstance(optimizer, torch.optim.LBFGS):
                prog = tqdm(total=optimizer.param_groups[0]['max_eval'])
            else:
                prog = tqdm(total=1)

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
            try:
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
            except KeyboardInterrupt:
                break

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
                    t0 = y[:, 0, measure_idx]
                    process.init_mean[0] = t0[~torch.isnan(t0) & ~torch.isinf(t0)].mean()

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
                start_offsets: Optional[Sequence] = None,
                out_timesteps: Optional[int] = None,
                initial_state: Union[Tensor, Tuple[Optional[Tensor], Optional[Tensor]]] = (None, None),
                every_step: bool = True,
                **kwargs) -> Predictions:
        """
        Generate n-step-ahead predictions from the model.

        :param input: A (group X time X measures) tensor. Optional if `initial_state` is specified.
        :param n_step: What is the horizon for the predictions output for each timepoint? Defaults to one-step-ahead
         predictions (i.e. n_step=1).
        :param start_offsets: If your model includes seasonal processes, then these needs to know the start-time for
         each group in ``input``. If you passed ``dt_unit`` when constructing those processes, then you should pass an
         array datetimes here. Otherwise you can pass an array of integers (or leave `None` if there are no seasonal
         processes).
        :param out_timesteps: The number of timesteps to produce in the output. This is useful when passing a tensor
         of predictors that goes later in time than the `input` tensor -- you can specify `out_timesteps=X.shape[1]` to
         get forecasts into this later time horizon.
        :param initial_state: The initial prediction for the state of the system: a tuple of tensors representing the
         initial mean and covariance. Default is ``(self.initial_mean, self.initial_covariance()``. You can pass your
         own for one or both of these, which can come from either a previous prediction or from a separate |nn.Module|
         that predicts the initial state.
        :param every_step: By default, `n_step` ahead predictions will be generated at every timestep. If
         `every_step=False`, then these predictions will only be generated every `n_step` timesteps. For example, with
         hourly data, `n_step=24` and every_step=True, each timepoint would be a forecast generated with data 24-hours
         in the past. But with `every_step=False` in this case, then the first timestep would be 1-step-ahead, the 2nd
         would be 2-step-ahead, ... the 23rd would be 24-step-ahead, the 24th would be 1-step-ahead, etc. The advantage
         to every_step=False is speed: training data for long-range forecasts can be generated without requiring the
         model to produce and discard intermediate predictions every timestep.
        :param kwargs: Further arguments passed to the `processes`. For example, many seasonal processes require a
         `state_datetimes` argument; the `LinearModel` and `NN` processes expect a `X` argument for predictors.
        :return: A :class:`Predictions` object with
         :func:`Predictions.log_prob()` and :func:`Predictions.to_dataframe()` methods.
        """

        if out_timesteps is None and input is None:
            raise RuntimeError("If `input` is None must specify `out_timesteps`")

        if isinstance(initial_state, Tensor):
            initial_state = (initial_state, None)
        initial_state = self._prepare_initial_state(
            initial_state,
            start_offsets=start_offsets,
            num_groups=None if input is None else input.shape[0]
        )

        means, covs, R, H = self._script_forward(
            input=input,
            initial_state=initial_state,
            n_step=n_step,
            every_step=every_step,
            out_timesteps=out_timesteps,
            **self._parse_design_kwargs(input=input, out_timesteps=out_timesteps or input.shape[1], **kwargs)
        )
        return Predictions(state_means=means, state_covs=covs, R=R, H=H, kalman_filter=self)

    @torch.jit.ignore
    def _prepare_initial_state(self,
                               initial_state,
                               start_offsets: Optional[Sequence] = None,
                               num_groups: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        init_mean, init_cov = initial_state
        if init_mean is None:
            init_mean = self.initial_mean[None, :]
        assert len(init_mean.shape) == 2

        if init_cov is None:
            # TODO: what about predicting with kwargs?
            init_cov = self.initial_covariance()[None, :]

        if num_groups is None and start_offsets is not None:
            num_groups = len(start_offsets)

        if num_groups is not None:
            assert init_mean.shape[0] in (num_groups, 1)
            init_mean = init_mean.expand(num_groups, -1)
            init_cov = init_cov.expand(num_groups, -1, -1)

        measure_scaling = torch.diag_embed(self._get_measure_scaling())
        init_cov = measure_scaling @ init_cov @ measure_scaling

        # seasonal processes need to offset the initial mean:
        init_mean_offset = []
        for pid, p in self.named_processes():
            _process_slice = slice(*self.process_to_slice[pid])
            init_mean_offset.append(p.offset_initial_state(init_mean[:, _process_slice], start_offsets))
        init_mean_offset = torch.cat(init_mean_offset, 1)

        return init_mean_offset, init_cov

    @torch.jit.export
    def _script_forward(self,
                        input: Optional[Tensor],
                        static_kwargs: Dict[str, Dict[str, Tensor]],
                        time_varying_kwargs: Dict[str, Dict[str, List[Tensor]]],
                        initial_state: Tuple[Tensor, Tensor],
                        n_step: int = 1,
                        out_timesteps: Optional[int] = None,
                        every_step: bool = True) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        :param input: A (group X time X measures) tensor. Optional if `initial_state` is specified.
        :param static_kwargs: Keyword-arguments to the Processes which do not vary over time.
        :param time_varying_kwargs: Keyword-arguments to the Process which do vary over time. At each timestep, each
        kwarg gets sliced for that timestep.
        :param initial_state: A (mean, cov) tuple to use as the initial state.
        :param n_step: What is the horizon for predictions? Defaults to one-step-ahead (i.e. n_step=1).
        :param out_timesteps: The number of timesteps in the output. Might be longer than input if forecasting.
        :param every_step: Experimental. When n_step>1, we can generate these n-step-ahead predictions at every
         timestep (e.g. 24-hour-ahead predictions every hour), in which case we'd save the 24-step-ahead prediction.
         Alternatively, we could generate 24-hour-ahead predictions at every 24th hour, in which case we'd save
         predictions 1-24. The former corresponds to every_step=True, the latter to every_step=False. If n_step=1
         (the default) then this option has no effect.
        :return: means, covs, R, H
        """
        assert n_step > 0

        mean1step, cov1step = initial_state

        if input is None:
            if out_timesteps is None:
                raise RuntimeError("If `input` is None must pass `out_timesteps`")
            inputs = []

            num_groups = mean1step.shape[0]
        else:
            if len(input.shape) != 3:
                raise ValueError(f"Expected len(input.shape) == 3 (group,time,measure)")
            if input.shape[-1] != len(self.measures):
                raise ValueError(f"Expected input.shape[-1] == {len(self.measures)} (len(self.measures))")

            num_groups = input.shape[0]
            if mean1step.shape[0] == 1:
                mean1step = mean1step.expand(num_groups, -1)
            if cov1step.shape[0] == 1:
                cov1step = cov1step.expand(num_groups, -1, -1)

            inputs = input.unbind(1)
            if out_timesteps is None:
                out_timesteps = len(inputs)

        Fs, Hs, Qs, Rs = self.build_design_mats(
            static_kwargs=static_kwargs,
            time_varying_kwargs=time_varying_kwargs,
            num_groups=num_groups,
            out_timesteps=out_timesteps
        )

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

    def build_design_mats(self,
                          static_kwargs: Dict[str, Dict[str, Tensor]],
                          time_varying_kwargs: Dict[str, Dict[str, List[Tensor]]],
                          num_groups: int,
                          out_timesteps: int) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
        # measure scaling
        measure_scaling = torch.diag_embed(self._get_measure_scaling())

        # process-variance:
        if 'process_covariance' in time_varying_kwargs and \
                self.process_covariance.expected_kwarg in time_varying_kwargs['process_covariance']:
            pvar_inputs = time_varying_kwargs['process_covariance'][self.process_covariance.expected_kwarg]
            Qs: List[Tensor] = []
            for t in range(out_timesteps):
                Qs.append(measure_scaling @ self.process_covariance(pvar_inputs[t]) @ measure_scaling)
        else:
            pvar_input: Optional[Tensor] = None
            if 'process_covariance' in static_kwargs and \
                    self.process_covariance.expected_kwarg in static_kwargs['process_covariance']:
                pvar_input = static_kwargs['process_covariance'].get(self.process_covariance.expected_kwarg)
            Q = measure_scaling @ self.process_covariance(pvar_input) @ measure_scaling
            if len(Q.shape) == 2:
                Q = Q.expand(num_groups, -1, -1)
            Qs = [Q] * out_timesteps

        # measure-variance:
        if 'measure_covariance' in time_varying_kwargs and \
                self.measure_covariance.expected_kwarg in time_varying_kwargs['measure_covariance']:
            mvar_inputs = time_varying_kwargs['measure_covariance'][self.measure_covariance.expected_kwarg]
            Rs: List[Tensor] = []
            for t in range(out_timesteps):
                Rs.append(self.measure_covariance(mvar_inputs[t]))
        else:
            mvar_input: Optional[Tensor] = None
            if 'measure_covariance' in static_kwargs and \
                    self.measure_covariance.expected_kwarg in static_kwargs['measure_covariance']:
                mvar_input = static_kwargs['measure_covariance'].get(self.measure_covariance.expected_kwarg)
            R = self.measure_covariance(mvar_input)
            if len(R.shape) == 2:
                R = R.expand(num_groups, -1, -1)
            Rs = [R] * out_timesteps

        # Transition and Measurement Mats:
        base_F = torch.zeros(
            (num_groups, self.state_rank, self.state_rank),
            dtype=measure_scaling.dtype,
            device=measure_scaling.device
        )
        base_H = torch.zeros(
            (num_groups, len(self.measures), self.state_rank),
            dtype=measure_scaling.dtype,
            device=measure_scaling.device
        )
        for pid, process1 in self.processes.items():
            p_tv_kwargs = time_varying_kwargs.get(pid)
            p_static_kwargs = static_kwargs.get(pid)
            _process_slice1 = slice(*self.process_to_slice[pid])

            # static H:
            if p_tv_kwargs is None or process1.h_kwarg not in p_tv_kwargs:
                h_input = None if p_static_kwargs is None else p_static_kwargs.get(process1.h_kwarg)
                ph = process1.h_forward(h_input)
                base_H[:, self.measure_to_idx[process1.measure], _process_slice1] = ph

            # static F:
            if p_tv_kwargs is None or process1.f_kwarg not in p_tv_kwargs:
                f_input = None if p_static_kwargs is None else p_static_kwargs.get(process1.h_kwarg)
                pf = process1.f_forward(f_input)
                base_F[:, _process_slice1, _process_slice1] = pf

        Fs: List[Tensor] = []
        Hs: List[Tensor] = []
        for t in range(out_timesteps):
            Ft = base_F.clone()
            Ht = base_H.clone()
            for pid, process2 in self.processes.items():
                p_tv_kwargs = time_varying_kwargs.get(pid)
                _process_slice2 = slice(*self.process_to_slice[pid])

                # tv H:
                if p_tv_kwargs is not None and process2.h_kwarg in p_tv_kwargs:
                    ph = process2.h_forward(p_tv_kwargs[process2.h_kwarg][t])
                    Ht[:, self.measure_to_idx[process2.measure], _process_slice2] = ph

                # tv F:
                if p_tv_kwargs is not None and process2.f_kwarg in p_tv_kwargs:
                    pf = process2.f_forward(p_tv_kwargs[process2.f_kwarg][t])
                    Ft[:, _process_slice2, _process_slice2] = pf
            Fs.append(Ft)
            Hs.append(Ht)
        return Fs, Hs, Qs, Rs

    @torch.no_grad()
    @torch.jit.ignore()
    def simulate(self,
                 out_timesteps: int,
                 initial_state: Tuple[Optional[Tensor], Optional[Tensor]] = (None, None),
                 start_offsets: Optional[Sequence] = None,
                 num_sims: Optional[int] = None,
                 progress: bool = False,
                 **kwargs):
        """
        Generate simulated state-trajectories from your model.

        :param out_timesteps: The number of timesteps to generate in the output.
        :param initial_state: The initial state of the system: a tuple of `mean`, `cov`.
        :param start_offsets: If your model includes seasonal processes, then these needs to know the start-time for
         each group in ``input``. If you passed ``dt_unit`` when constructing those processes, then you should pass an
         array datetimes here. Otherwise you can pass an array of integers (or leave `None` if there are no seasonal
         processes).
        :param num_sims: The number of state-trajectories to simulate.
        :param progress: Should a progress-bar be displayed? Requires `tqdm`.
        :param kwargs: Further arguments passed to the `processes`.
        :return: A :class:`Simulations` object with a :func:`Simulations.sample()` method.
        """

        design_kwargs = self._parse_design_kwargs(input=None, out_timesteps=out_timesteps, **kwargs)

        mean, cov = self._prepare_initial_state(initial_state, start_offsets=start_offsets, num_groups=num_sims)

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

        Fs, Hs, Qs, Rs = self.build_design_mats(num_groups=num_sims, out_timesteps=out_timesteps, **design_kwargs)

        dist_cls = self.kf_step.get_distribution()

        means: List[Tensor] = []
        for t in times:
            mean = dist_cls(mean, cov).rsample()
            mean, cov = self.kf_step.predict(mean, .0001 * torch.eye(mean.shape[-1]), F=Fs[t], Q=Qs[t])
            means.append(mean)

        return Simulations(torch.stack(means, 1), H=torch.stack(Hs, 1), R=torch.stack(Rs, 1), kalman_filter=self)

    @torch.jit.ignore()
    def _parse_design_kwargs(self, input: Optional[Tensor], out_timesteps: int, **kwargs) -> Dict[str, dict]:
        static_kwargs = defaultdict(dict)
        time_varying_kwargs = defaultdict(dict)
        unused = set(kwargs)
        kwargs.update(input=input, current_timestep=torch.tensor(list(range(out_timesteps))).view(1, -1, 1))
        for submodule_nm, submodule in list(self.named_processes()) + list(self.named_covariances()):
            for found_key, key_name, value in submodule.get_kwargs(kwargs):
                unused.discard(found_key)
                if value is not None and len(value.shape) == 3:
                    time_varying_kwargs[submodule_nm][key_name] = value.unbind(1)
                elif value is not None:
                    assert len(value.shape) <= 2, f"'{found_key}' has unexpected ndim {len(value.shape)}, expected <= 3"
                    static_kwargs[submodule_nm][key_name] = value

        if unused:
            warn(f"There are unused keyword arguments:\n{unused}")
        return {
            'static_kwargs': dict(static_kwargs),
            'time_varying_kwargs': dict(time_varying_kwargs)
        }

    def _get_measure_scaling(self) -> Tensor:
        mcov = self.measure_covariance(None, _ignore_input=True)
        if self._scale_by_measure_var:
            measure_var = mcov.diagonal(dim1=-2, dim2=-1)
            multi = torch.zeros(mcov.shape[0:-2] + (self.state_rank,), dtype=mcov.dtype, device=mcov.device)
            for pid, process in self.processes.items():
                pidx = self.process_to_slice[pid]
                multi[..., slice(*pidx)] = measure_var[..., self.measure_to_idx[process.measure]].sqrt().unsqueeze(-1)
            assert (multi > 0).all()
        else:
            multi = torch.ones((self.state_rank,), dtype=mcov.dtype, device=mcov.device)
        return multi

    def __repr__(self) -> str:
        return f'{type(self).__name__}' \
               f'(processes={repr(list(self.processes.values()))}, measures={repr(list(self.measures))})'
