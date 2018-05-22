from torch_kalman.design.nn_output import NNOutputTracker
from torch_kalman.utils.torch_utils import expand


class DesignMatrix(NNOutputTracker):
    def __init__(self):
        self._template = None
        self.batch_cache = {}
        super().__init__()

    @property
    def template(self):
        raise NotImplementedError()

    def register_variables(self):
        raise NotImplementedError()

    def reset(self):
        self._template = None
        self.batch_cache = {}

    def create_for_batch(self, time, **kwargs):
        bs = kwargs['kf_input'].data.shape[0]

        if self.nn_module.isnull:
            return expand(self.template, bs)
        else:
            # check kwargs:
            missing_kwargs = self.input_names - set(kwargs.keys())
            if len(missing_kwargs) == 0:
                nn_module_kwargs = {argname: kwargs[argname] for argname in self.input_names}
            else:
                raise TypeError("missing {} required arguments: {}".format(len(missing_kwargs), missing_kwargs))

            # expand, replacing NNOutput placeholders:
            nn_outputs = self.nn_module(time=time, **nn_module_kwargs)
            expanded = expand(self.template, bs).clone()
            for (row, col), output in nn_outputs:
                expanded[:, row, col] = output
            return expanded
