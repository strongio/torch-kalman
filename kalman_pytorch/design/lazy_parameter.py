import torch


class LazyParameter(object):
    def __init__(self, parameter):
        self.parameter = parameter
        self.lambda_chain = []
        self.after_init_idx = 0

    def add_lambda(self, lam):
        self.lambda_chain.append(lam)

    def with_added_lambda(self, lam):
        new_lazy = self.__class__(parameter=self.parameter)
        for old_lam in self.lambda_chain[self.after_init_idx:]:
            new_lazy.add_lambda(old_lam)
        new_lazy.add_lambda(lam)
        return new_lazy

    def __call__(self):
        param_out = self.parameter
        for lam in self.lambda_chain:
            param_out = lam(param_out)
        return param_out


class LogLinked(LazyParameter):
    def __init__(self, parameter):
        super(LogLinked, self).__init__(parameter=parameter)
        self.lambda_chain.append(torch.exp)
        self.after_init_idx = 1
