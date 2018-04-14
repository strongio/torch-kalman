class NNInput(object):
    def __init__(self, name):
        self.name = name

    def slice(self, tensor, time):
        raise NotImplementedError()


class CurrentTime(NNInput):
    def __init__(self, name, num_dims=3):
        super().__init__(name=name)
        assert num_dims == 2 or num_dims == 3
        self.num_dims = num_dims

    def slice(self, tensor, time):
        if self.num_dims == 3:
            return tensor[:, time, :]
        else:
            return tensor[:, time]


class UpToCurrentTime(NNInput):
    def __init__(self, name, num_dims=3):
        super().__init__(name=name)
        assert num_dims == 2 or num_dims == 3
        self.num_dims = num_dims

    def slice(self, tensor, time):
        if self.num_dims == 3:
            return tensor[:, 0:time, :]
        else:
            return tensor[:, 0:time]
