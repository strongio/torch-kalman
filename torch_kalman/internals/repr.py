from torch import Tensor


class NiceRepr:
    _repr_attrs = None

    def __repr__(self) -> str:
        kwargs = []
        for k in self._repr_attrs:
            v = getattr(self, k)
            if isinstance(v, Tensor):
                v = v.size()
            kwargs.append("{}={!r}".format(k, v))
        return "{}({})".format(type(self).__name__, ", ".join(kwargs))
