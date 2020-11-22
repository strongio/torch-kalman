from typing import Sequence, Dict
from warnings import warn

import torch
from torch.nn import Embedding
import numpy as np


class NamedEmbedding(Embedding):
    """
    Designed for use in the `KalmanFilter` measure/process_var_nns (the "per_group" alias). Similar to
    torch.nn.Embedding, but takes any sequence of names instead of a tensor of integers. This restricts the input to
    1D, but it is convenient as the names are stored in the state_dict (so e.g. model-state is robust to the order in
    which the names are passed at training vs. evaluation).
    ```
    # data
    df = pd.concat([
        pd.DataFrame({'value' : v + np.random.normal(size=10)}).assign(group=g)
        for v,g in enumerate(['A','B','C','D'])
    ])
    df['group'] = df['group'].astype(pd.CategoricalDtype())

    # nn:
    nn = NamedEmbedding(4, 1, dev_code=True)

    # include bias:
    nn.bias = torch.nn.Parameter(torch.zeros(1))
    nn.register_parameter('bias',nn.bias)

    # train:
    nn.opt = torch.optim.Adam(nn.parameters(), lr=.1)
    for i in range(500):
        nn.opt.zero_grad()
        pred = nn(df['group'].tolist()) + nn.bias
        losses = (pred.squeeze() - torch.from_numpy(df['value'].values))**2
        loss = losses.mean()
        loss.backward()
        nn.opt.step()

    # nn learned group means
    with torch.no_grad():
        print(pred.unique())
    print(df.groupby('group')['value'].mean())

    # unknown groups at eval default to bias:
    nn.eval()
    print(df['value'].mean())
    print(nn(['E']) + nn.bias)
    ```
    """
    max_name_len = 64

    def __init__(self, num_embeddings: int, embedding_dim: int, dev_code: bool = False, **kwargs):
        """
        :param num_embeddings: The size of the dictionary of embeddings.
        :param embedding_dim: The size of each embedding.
        :param dev_code: If True, then the actual number of weights will be reduced by 1, and the first entry in the
        dictionary will not get its own vector (instead, its vector will be -sum(all_other_vectors)). This is useful if
        the `NamedEmbedding` is being used as part of a larger network where the bias/intercept is already being
        learned. In that case, each embedding will represent that entry's deviation from the average, and when
        predicting for new entries that were not trained on, the average embedding will be output from `forward()`.
        Please note that `dev_code=True` also overrides the default `reset_parameters()` method to intialize weights to
        zero.
        :param kwargs: Other keyword-arguments passed to torch.nn.Embedding.
        """
        self.dev_code = dev_code

        super().__init__(
            num_embeddings=num_embeddings - int(dev_code),
            embedding_dim=embedding_dim,
            padding_idx=0 if dev_code else None,
            **kwargs
        )

        self._names_as_ints = torch.nn.Parameter(
            torch.zeros((self.num_embeddings + int(self.dev_code), self.max_name_len))
        )
        self._names_as_ints.requires_grad_(False)
        self._name_to_idx = None

    @property
    def name_to_idx(self) -> Dict[str, int]:
        """
        We'd like to store `name_to_idx` in the state so that it's saved, and loaded with load_state_dict. But the
        state-dict is only for Parameters -- and we can't use a ParameterDict for the names because load_state_dict
        will complain about mismatched keys. So we convert the names to integers then save them in a 2d tensor.
        """
        if self._name_to_idx is None:
            self._name_to_idx = {}
            for i, int_repr in enumerate(self._names_as_ints.tolist()):
                if set(int_repr) == {0.0}:
                    # unassigned
                    continue
                self._name_to_idx[_ints_to_str(int_repr).rstrip()] = i
        return self._name_to_idx

    def _input_to_idx(self, input: Sequence[str]) -> torch.Tensor:
        num_idx = self.num_embeddings + int(self.dev_code)
        indices = []
        for nm in input:
            nm = str(nm).rstrip()
            if nm not in self.name_to_idx:
                if not self.training:
                    indices.append(-1)
                    continue
                if len(self.name_to_idx) >= num_idx:
                    raise RuntimeError(
                        f"Got a new group name '{nm}', but all {num_idx} idx are taken by previous "
                        f"groups. If trying to predict for new groups set `self.train(False)`"
                    )
                else:
                    self._names_as_ints[len(self.name_to_idx)] = torch.tensor(_str_to_ints(nm, width=self.max_name_len))
                    self._name_to_idx = None  # reset `name_to_idx` cache
            indices.append(self.name_to_idx[nm])

        with torch.no_grad():
            idx = torch.tensor(indices, dtype=torch.long)
        return idx

    def forward(self, input: Sequence[str]) -> torch.Tensor:
        idx = self._input_to_idx(input)
        out = torch.zeros((len(input), self.embedding_dim))
        in_embedding = (idx > 0)
        if not self.dev_code:
            in_embedding |= (idx == 0)
        if in_embedding.any():
            out[np.where(in_embedding)] = super().forward(idx[np.where(in_embedding)] - int(self.dev_code))
        if self.dev_code:
            out[np.where(idx == 0)] = -torch.sum(self.weight, 0)
        return out

    def reset_parameters(self):
        super().reset_parameters()
        if self.dev_code:
            self.weight.data *= 0.0


def _str_to_ints(astr: str, width: int) -> Sequence[int]:
    if len(astr) > width:
        warn(f"'{astr}' has more characters than {width}, will be truncated")
    return [ord(astr[i]) if i < len(astr) else 32 for i in range(width)]


def _ints_to_str(ints: Sequence) -> str:
    return "".join(str(chr(int(x))) for x in ints)
