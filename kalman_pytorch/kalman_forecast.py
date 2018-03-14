from kalman_pytorch.kalman_filter import KalmanFilter

import torch
from torch.nn import Parameter
from torch.autograd import Variable

import numpy as np

from utils.torch_utils import log_std_to_var, expand


class KalmanForecast(KalmanFilter):
    def __init__(self, num_variables, num_seasons=0, common=None):
        super(KalmanForecast, self).__init__()

        self.num_variables = num_variables
        self.num_seasons = num_seasons
        self.common = common
        self._var_indices = self.var_indices

        self.rank = self.num_variables * (self.num_seasons + 2)
        if self.common == 'trend':
            self.rank += 2
        elif self.common == 'level':
            self.rank += 1
        elif self.common is not None:
            raise Exception("`common` must be one of (None, 'level', 'trend')")

        # parameters:
        # initial mean, covariance:
        self.initial_x = Parameter(torch.randn(self.rank))
        self.initial_log_std = Parameter(torch.randn(self.rank))

        # observation noise:
        self.log_obs_std = Parameter(torch.randn(self.num_variables))
        self.error_cor_logis = Parameter(torch.zeros(1))

        # process-noise for underlying level/trend:
        n_lt = self.num_variables if self.common is None else (self.num_variables + 1)
        self.log_trend_std = Parameter(torch.randn(n_lt))

        # process-noise for underlying seasonality:
        if self.num_seasons > 0:
            self.log_season_std = Parameter(torch.randn(self.num_variables))

    def initializer(self, tens):
        """
        :param tens: A tensor of observed values.
        :return: Initial values for mean, cov
        """
        initial_mean = self.initial_x[:, None]
        initial_cov = log_std_to_var(self.initial_log_std) * Variable(torch.eye(self.rank, self.rank))
        bs = tens.data.shape[0]
        return expand(initial_mean, bs), expand(initial_cov, bs)

    @property
    def var_indices(self):
        """
        This is used for a cumbersome task that's required for creating design-matrices that pytorch can optimize:
        specifying the mapping of optimizable parameters to positions in these design matrices.

        :return: Three dictionaries are returned:
          (1) Each key is a variable, each value contains the indices of the 'level' components for that variable.
          (2) Each key is a variable, each value contains the indices of the 'trend' components for that variable.
          (3) Each key is a variable, each value contains the indices of the 'season' components for that variable.
        """
        if getattr(self, '_var_indices', None) is None:
            i = 0
            indexer = {'season': {}, 'trend': {}, 'level': {}}
            for v in range(self.num_variables):
                indexer['trend'][v] = []
                for _ in range(2):
                    indexer['trend'][v].append(i)
                    i += 1
                if self.num_seasons > 0:
                    indexer['season'][v] = []
                for _ in range(self.num_seasons):
                    indexer['season'][v].append(i)
                    i += 1
            if self.common == 'trend':
                indexer['trend']['common'] = [i, i + 1]
            elif self.common == 'level':
                indexer['level']['common'] = [i]
            self._var_indices = indexer['level'], indexer['trend'], indexer['season']

        return self._var_indices

    def destroy_design_mats(self):
        self._Q = None
        self._F = None
        self._H = None
        self._R = None

    @property
    def R(self):
        if self._R is None:
            # cor
            cor_mat = Variable(torch.zeros(self.num_variables, self.num_variables))
            triu_mask = torch.triu(torch.ones(*cor_mat.data.shape), 1) > 0
            cor_mat[triu_mask] = torch.sigmoid(self.error_cor_logis).repeat(torch.sum(triu_mask))
            tril_mask = torch.tril(torch.ones(*cor_mat.data.shape), -1) > 0
            cor_mat[tril_mask] = torch.sigmoid(self.error_cor_logis).repeat(torch.sum(tril_mask))
            cor_mat[torch.diag(torch.ones(self.num_variables)) > 0] = 1.0

            # sigma
            sigma_mat = torch.diag(log_std_to_var(self.log_obs_std))

            # quad_form_diag
            self._R = torch.mm(torch.mm(sigma_mat, cor_mat), sigma_mat)

        return self._R

    @property
    def Q(self):
        if self._Q is None:
            Q = Variable(torch.zeros(self.rank, self.rank))
            level_indices, trend_indices, season_indices = self.var_indices

            # level:
            for variable, indexer in level_indices.iteritems():
                pidx = self.num_variables if variable == 'common' else variable  # index after last var
                Q[indexer[0], indexer[0]] = log_std_to_var(self.log_trend_std[pidx])

            # trend uses discrete_white_noise approximation, [[0.25, 0.50],[0.50, 1.00]]
            for variable, indexer in trend_indices.iteritems():
                pidx = self.num_variables if variable == 'common' else variable  # index after last var
                variance = log_std_to_var(self.log_trend_std[pidx])
                Q[indexer[0], indexer[0]] = 0.25 * variance
                Q[indexer[0], indexer[1]] = 0.50 * variance
                Q[indexer[1], indexer[0]] = 0.50 * variance
                Q[indexer[1], indexer[1]] = 1.00 * variance

            # seasonality:
            for variable, indexer in season_indices.iteritems():
                if variable == 'common':
                    raise NotImplementedError("Common seasonality not implemented.")
                Q[indexer[0], indexer[0]] = log_std_to_var(self.log_season_std[variable])
            self._Q = Q

        return self._Q

    @property
    def H(self):
        if self._H is None:
            H = np.zeros((self.num_variables, self.rank))

            level_indices, trend_indices, season_indices = self.var_indices

            for to_var in range(self.num_variables):
                # level
                for from_var in [to_var, 'common']:
                    idx = level_indices.get(from_var, None)
                    if idx is not None:
                        H[to_var, idx] = 1

                # trend
                for from_var in [to_var, 'common']:
                    idx = trend_indices.get(from_var, None)
                    if idx is not None:
                        H[to_var, idx] = [1, 0]

                # season
                for from_var in [to_var, 'common']:
                    idx = season_indices.get(from_var, None)
                    if idx is not None:
                        H[to_var, idx] = [1] + [0] * (self.num_seasons - 1)

            self._H = Variable(torch.from_numpy(H).float())

        return self._H

    @property
    def F(self):
        if self._F is None:
            F = np.zeros((self.rank, self.rank))
            level_indices, trend_indices, season_indices = self.var_indices

            # level
            for indexer in level_indices.values():
                F[indexer[0], indexer[0]] = 1.0

            # trend
            for indexer in trend_indices.values():
                F[indexer, indexer] = 1.0  # diagonal
                F[indexer[0], indexer[1]] = 1.0  # 1st row second col

            # seasonality:
            for indexer in season_indices.values():
                F[indexer[0], indexer[:-1]] = [-1.0] * (self.num_seasons - 1)
                F[indexer[1:], indexer[:-1]] = 1.0

            self._F = Variable(torch.from_numpy(F).float())

        return self._F
