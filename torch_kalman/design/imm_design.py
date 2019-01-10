from collections import OrderedDict
from typing import Optional, Iterable, Sequence, Dict, Generator

import torch
from torch import Tensor
from torch.nn import Parameter, ParameterList

from torch_kalman.design import Design
from torch_kalman.design.for_batch import DesignForBatch
from torch_kalman.process import Process
from torch_kalman.utils import itervalues_sorted_keys


class IMMModel:
    def __init__(self, id: str):
        self.id = id

    def parameters(self) -> Generator[Parameter, None, None]:
        raise NotImplementedError


class SimpleIMMModel(IMMModel):
    """
    Simplified IMM that only allows other 'models' to be modified versions of the base model, but with a multiplicative
    adjustment to the process variance.
    """

    def __init__(self, id: str):
        super().__init__(id=id)
        self.modifications = {}

    def parameters(self) -> Generator[Parameter, None, None]:
        for param in itervalues_sorted_keys(self.modifications):
            yield param

    # def add_proc_var_mod(self,
    #                      process: Process,
    #                      state_elements: Optional[Sequence[str]] = None,
    #                      init_offset: float = 0.) -> None:
    #
    #     if state_elements is None:
    #         state_elements = process.dynamic_state_elements
    #     else:
    #         for se in state_elements:
    #             assert se in process.dynamic_state_elements, f"'{se}' not a dynamic-state-element in '{process.id}'"
    #
    #     for state_element in state_elements:
    #         self.modifications[(process.id, state_element)] = Parameter(init_offset + .01 * torch.randn(1))
    #
    # def modify_base_process_cov(self, design: Design, base_mat: Tensor) -> Tensor:
    #     pse_idx = {pse: i for i, pse in enumerate(design.all_state_elements())}
    #
    #     model_proc_cov = base_mat.clone()
    #     diag_multi = torch.eye(model_proc_cov.shape[-1], device=self.device)
    #     for (process, state_element), param in self.modifications.items():
    #         idx = pse_idx[(process, state_element)]
    #         diag_multi[idx, idx] = param.exp()
    #
    #     return diag_multi.matmul(model_proc_cov).matmul(diag_multi)


class IMMDesign(Design):
    def __init__(self,
                 processes: Iterable[Process],
                 measures: Iterable[str],
                 models: Iterable[IMMModel],
                 device: Optional[torch.device] = None):

        super().__init__(processes=processes, measures=measures, device=device)

        self.models = OrderedDict()
        for model in models:
            self.models[model.id] = model

        self._mode_prob_params = Parameter(torch.randn(self.num_models - 1, device=self.device))
        self._transition_prob_params = \
            ParameterList(parameters=[Parameter(torch.randn(self.num_models - 1, device=self.device))
                                      for _ in range(self.num_models)])

    @property
    def mode_probs(self):
        if self.num_models == 1:
            return torch.ones(1)
        elif self.num_models != 2:
            raise NotImplementedError("Please report this error to the package maintainer.")
        probs = torch.zeros(2, device=self.device)
        probs[0] = 1 - self._mode_prob_params.sigmoid()
        probs[1] = self._mode_prob_params.sigmoid()
        return probs

    @property
    def transition_probs(self):
        if self.num_models == 1:
            return torch.ones((1, 1))
        elif self.num_models != 2:
            raise NotImplementedError("Please report this error to the package maintainer.")
        probs = torch.zeros((2, 2))
        for r, param in enumerate(self._transition_prob_params):
            probs[r, 0] = 1 - param.sigmoid()
            probs[r, 1] = param.sigmoid()
        return probs

    def parameters(self):
        yield from super().parameters()
        yield self._mode_prob_params
        yield self._transition_prob_params
        for model in self.models.values():
            yield from model.parameters()

    @property
    def num_models(self) -> int:
        # self.models doesn't include the base model
        return len(self.models) + 1

    def for_batch(self, num_groups: int, num_timesteps: int, **kwargs):
        return IMMDesignForBatch(design=self,
                                 num_groups=num_groups,
                                 num_timesteps=num_timesteps,
                                 **kwargs)


class IMMDesignForBatch(DesignForBatch):
    def __init__(self,
                 design: 'IMMDesign',
                 num_groups: int,
                 num_timesteps: int,
                 **kwargs):
        self.design: IMMDesign = None
        super().__init__(design=design, num_groups=num_groups, num_timesteps=num_timesteps, **kwargs)
        self.num_models = design.num_models

    def F_init(self):
        super().F_init()
        # add a leading dimension for model:
        self.F_base = self.F_base.expand(self.num_models, -1, -1, -1).clone()

    # def init_process_cov_mats(self):
    #     proc_cov_model_0 = super().init_process_cov_mats()
    #
    #     proc_cov_all_models = [proc_cov_model_0]
    #     for model in self.design.models.values():
    #         proc_cov_all_models.append(model.modify_base_process_cov(self.design, proc_cov_model_0))
    #
    #     return torch.stack(proc_cov_all_models)

    # def F(self, t: int) -> torch.Tensor:
    #     if self._F is None:
    #         self._F = self.init_transition_mats()
    #
    #     if not self._F['dynamic_assignments']:
    #         return self._F['base']
    #     else:
    #         mat = self._F['base'].clone()
    #         for (r, c), values in self._F['dynamic_assignments']:
    #             mat[:, :, r, c] = values[t]
    #         return mat
    #
    # def init_transition_mats(self):
    #     mat_info = super().init_transition_mats()
    #     mat_info['base'] = mat_info['base'].expand(self.num_models, -1, -1, -1)
    #     return mat_info
