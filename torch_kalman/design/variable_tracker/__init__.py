"""
VariableTrackers keep track of where torch Parameters/Variables are located inside of other, destination Variables
(usually, design-matrices).

NNIOTrackers additionally keep track of where nn-module ouputs go in the destination matrix. Each NNOutput is associated with
 a nn.module, and NNIOTrackers allow the user to specify the inputs to each of these nn.modules, which are then unified into
a single nn.module using NNModuleConcatenator. The end result is that `DesignMatrix`s and other classes inheriting from
NNIOTracker can have nn-modules that fill their elements.
"""


class VariableTracker(object):
    def __init__(self):
        self.register_variables()

    def register_variables(self):
        raise NotImplementedError()
