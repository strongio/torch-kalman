from typing import TypeVar

import torch
from torch.nn import Parameter, ParameterList

from torch_kalman.design.imm_design import SimpleIMMDesign, SimpleIMMDesignForBatch
from torch_kalman.kalman_filter import KalmanFilter
from torch_kalman.state_belief import Gaussian
from torch_kalman.state_belief.families.imm_belief import IMMBelief


class IMM(KalmanFilter):
    def __init__(self, design: SimpleIMMDesign):
        super().__init__(design=design)

        self._mode_prob_params = Parameter(torch.randn(self.design.num_models - 1, device=self.design.device))
        self._transition_prob_params = \
            ParameterList(parameters=[Parameter(torch.randn(self.design.num_models - 1, device=self.design.device))
                                      for _ in range(self.design.num_models)])

    @property
    def mode_probs(self):
        if self.design.num_models == 1:
            return torch.ones(1)
        elif self.design.num_models != 2:
            raise NotImplementedError("Please report this error to the package maintainer.")
        probs = torch.zeros(2, device=self.design.device)
        probs[0] = 1 - self._mode_prob_params.sigmoid()
        probs[1] = self._mode_prob_params.sigmoid()
        return probs

    @property
    def transition_probs(self):
        if self.design.num_models == 1:
            return torch.ones((1, 1))
        elif self.design.num_models != 2:
            raise NotImplementedError("Please report this error to the package maintainer.")
        probs = torch.zeros((2, 2))
        for r, param in enumerate(self._transition_prob_params):
            probs[r, 0] = 1 - param.sigmoid()
            probs[r, 1] = param.sigmoid()
        return probs

    def _init_design(self, design: SimpleIMMDesign):
        self.design = design

    @property
    def family(self) -> TypeVar('IMMBelief'):
        return IMMBelief

    def predict_initial_state(self, design_for_batch: 'SimpleIMMDesignForBatch'):
        state_beliefs = [Gaussian(means=design_for_batch.initial_mean, covs=design_for_batch.initial_covariance)
                         for _ in range(design_for_batch.num_models)]
        return self.family(state_beliefs=state_beliefs,
                           mode_probs=self.mode_probs,
                           transition_probs=self.transition_probs)
