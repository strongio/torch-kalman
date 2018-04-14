class Process(object):
    def __init__(self, states):
        self.states = tuple(states)

    @property
    def observable(self):
        raise NotImplementedError()
