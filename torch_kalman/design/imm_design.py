from collections import OrderedDict
from typing import Optional, Iterable, Sequence, Dict

import torch
from torch.nn import Parameter

from torch_kalman.design import Design
from torch_kalman.design.for_batch import DesignForBatch
from torch_kalman.process import Process
from torch_kalman.utils import itervalues_sorted_keys


class SimpleIMMDesign(Design):
    """
    Simplified IMM that only allows other 'models' to be modified versions of the base model, but with some process variances
     multiplied by a parameter.
    """

    def __init__(self,
                 processes: Iterable[Process],
                 measures: Iterable[str],
                 device: Optional[torch.device] = None):

        super().__init__(processes=processes, measures=measures, device=device)

        self.models = OrderedDict()
        self._locked = False

    @property
    def num_models(self) -> int:
        self._locked = True
        return len(self.models) + 1

    def add_model(self, name: str):
        assert not self._locked, "Design is locked, can no longer add models (happens when accessing parameters/num_models)."
        self.models[name] = {}

    def add_process_mod(self,
                        model_name: str,
                        process_name: str,
                        state_elements: Optional[Sequence[str]] = None,
                        init_offset: float = 0.):
        assert not self._locked, "Design is locked, can no longer add models (happens when accessing parameters/num_models)."

        proc = self.processes[process_name]
        if state_elements is None:
            state_elements = proc.dynamic_state_elements
        else:
            for se in state_elements:
                assert se in proc.dynamic_state_elements, f"'{se}' not a dynamic-state-element in '{proc.id}'"

        for state_element in state_elements:
            self.models[model_name][(proc.id, state_element)] = Parameter(init_offset + .01 * torch.randn(1))

    def parameters(self):
        self._locked = True
        yield from super().parameters()
        for pars in self.models.values():
            for par in itervalues_sorted_keys(pars):
                yield par

    def for_batch(self, num_groups: int, num_timesteps: int, **kwargs):
        return SimpleIMMDesignForBatch(design=self,
                                       num_groups=num_groups,
                                       num_timesteps=num_timesteps,
                                       **kwargs)


class SimpleIMMDesignForBatch(DesignForBatch):
    def __init__(self,
                 design: 'SimpleIMMDesign',
                 num_groups: int,
                 num_timesteps: int,
                 **kwargs):
        super().__init__(design=design, num_groups=num_groups, num_timesteps=num_timesteps, **kwargs)
        self.num_models = design.num_models

    def _init_process_cov_mats(self, design: 'SimpleIMMDesign'):
        super()._init_process_cov_mats(design=design)

        if self._design_mat_time_mods['Q']:
            raise RuntimeError("Please report error to package maintainer.")

        pse_idx = {pse: i for i, pse in enumerate(design.all_state_elements())}

        new_proc_cov = [self._design_mat_bases['Q']]
        for updates in design.models.values():
            model_proc_cov = self._design_mat_bases['Q'].clone()
            diag_multi = torch.eye(model_proc_cov.shape[-1], device=self.device)
            for (process, state_element), param in updates.items():
                idx = pse_idx[(process, state_element)]
                diag_multi[idx, idx] = param.exp()
            model_proc_cov = model_proc_cov.matmul(diag_multi).matmul(model_proc_cov)
            new_proc_cov.append(model_proc_cov)

        self._design_mat_bases['Q'] = torch.stack(new_proc_cov)

    def F(self, t: int) -> torch.Tensor:
        F = super().F(t=t)
        return F.expand(self.num_models, -1, -1, -1)
