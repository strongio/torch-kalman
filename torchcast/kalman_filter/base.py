from collections import defaultdict
from typing import Tuple, List, Optional, Sequence, Dict, Iterable
from warnings import warn

import torch
from torch import nn, Tensor

from torchcast.covariance import Covariance
from torchcast.kalman_filter.gaussian import GaussianStep
from torchcast.kalman_filter.jit import ScriptKalmanFilter
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
    script_cls = ScriptKalmanFilter
    kf_step = GaussianStep

    def __init__(self,
                 processes: Sequence[Process],
                 measures: Sequence[str],
                 process_covariance: Optional[Covariance] = None,
                 measure_covariance: Optional[Covariance] = None,
                 initial_covariance: Optional[Covariance] = None,
                 compiled: bool = True,
                 **kwargs):
        super(KalmanFilter, self).__init__()

        if isinstance(measures, str):
            measures = [measures]
            warn(f"`measures` should be a list of strings not a string; interpreted as `{measures}`.")

        self._validate(processes, measures)

        # covariances:
        if process_covariance is None:
            process_covariance = Covariance.for_processes(processes, cov_type='process')

        if measure_covariance is None:
            measure_covariance = Covariance.for_measures(measures)

        if initial_covariance is None:
            initial_covariance = Covariance.for_processes(processes, cov_type='initial')

        self.script_module = self.script_cls(
            kf_step=self.kf_step(),
            processes=processes,
            measures=measures,
            process_covariance=process_covariance,
            measure_covariance=measure_covariance,
            initial_covariance=initial_covariance,
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

        means, covs, R, H = self.script_module(
            input=input,
            initial_state=initial_state,
            n_step=n_step,
            every_step=every_step,
            out_timesteps=out_timesteps,
            _disable_cache=kwargs.pop('_disable_cache', False),
            **self._parse_design_kwargs(input=input, out_timesteps=out_timesteps or input.shape[1], **kwargs)
        )
        return Predictions(state_means=means, state_covs=covs, R=R, H=H, kalman_filter=self)

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
                    try:
                        from tqdm.auto import tqdm
                        progress = tqdm
                    except ImportError:
                        warn("`progress=True` requires package `tqdm`.")
                        progress = lambda x: x
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
