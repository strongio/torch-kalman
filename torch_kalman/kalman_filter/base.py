from collections import defaultdict
from typing import Tuple, List, Optional, Sequence, Dict, Iterable
from warnings import warn

import torch
from torch import nn, Tensor

from torch_kalman.covariance import Covariance
from torch_kalman.kalman_filter.gaussian import GaussianStep
from torch_kalman.kalman_filter.jit import ScriptKalmanFilter
from torch_kalman.kalman_filter.predictions import Predictions
from torch_kalman.kalman_filter.simulations import Simulations
from torch_kalman.process.regression import Process


class KalmanFilter(nn.Module):
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
        """
        :param processes: A list of `Process` modules.
        :param measures: A list of strings specifying the names of the dimensions of the time-series being measured.
        :param process_covariance: A module created with `Covariance.from_processes(processes, cov_type='process')`.
        :param measure_covariance: A module created with `Covariance.from_measures(measures)`.
        :param initial_covariance: A module created with `Covariance.from_processes(processes, cov_type='initial')`.
        :param compiled: Should the core modules be passed through torch.jit.script to compile them to TorchScript?
        Can be disabled if compilation issues arise.
        :param kwargs: Further arguments passed to ScriptKalmanFilter's child-classes (base-class takes no kwargs).
        """
        super(KalmanFilter, self).__init__()

        self._validate(processes, measures)

        # covariances:
        if process_covariance is None:
            process_covariance = Covariance.from_processes(processes, cov_type='process')

        if measure_covariance is None:
            measure_covariance = Covariance.from_measures(measures)

        if initial_covariance is None:
            initial_covariance = Covariance.from_processes(processes, cov_type='initial')

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
