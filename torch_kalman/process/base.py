from typing import Generator, Sequence, Dict, Union, Tuple

from torch import Tensor
from torch.nn import Parameter

from torch_kalman.covariance import Covariance
from torch_kalman.process.for_batch import ProcessForBatch


class Process:
    def __init__(self,
                 id: str,
                 state_elements: Sequence[str],
                 transitions: Dict[str, Dict[str, Union[float, None]]]):
        """

        :param id: A unique string identifying the process.
        :param state_elements: A sequence of names for the states that make up this process.
        :param transitions: A dictionary whose structure is easier to understand by example. For example:
            `transitions=dict(position = dict(position = 1.0, velocity = 1.0))`
        is designed to be read like an equation:
            position_t2 = 1.0 * position_t1 + 1.0 * velocity_t1
        That is, we specify a Dict whose keys are states being transitioned *to*, and whose values are themselves Dicts
        specifying the states *from* which we are transitioning. When specifying None for a value, this means that the value
        needs to be generated on a per-batch basis, via ProcessForBatch.set_transition.
        """

        self.id = str(id)
        self._state_element_idx = None

        # measures:
        self.state_elements_to_measures = {}

        # state elements:
        assert len(state_elements) == len(set(state_elements)), "Duplicate `state_elements` now allowed."
        self.state_elements = state_elements

        # transitions:
        self.transitions = transitions
        for to_el, from_els in self.transitions.items():
            assert to_el in self.state_elements, f"`{to_el}` is in transitions but not state_elements"
            for from_el in from_els.keys():
                assert to_el in self.state_elements, f"`{from_el}` is in transitions but not state_elements"

        # expected kwargs
        self.expected_batch_kwargs = ()

        # if no per-batch modification, can avoid repeated computations:
        self.F_base = None

    def link_to_design(self, design: 'Design') -> None:
        """
        Some processes need to know about the design they're nested within (e.g., a seasonal process whose characteristics
        depend on another seasonal process it is nested within). This method add the necessary information to the process.

        :param design: The design this process is embedded in.
        """
        pass

    def requires_grad_(self, requires_grad):
        for param in self.parameters():
            param.requires_grad_(requires_grad=requires_grad)

    @property
    def requires_grad(self):
        return any(param.requires_grad for param in self.parameters())

    def measures(self):
        return set(measure for measure, _ in self.state_elements_to_measures.keys())

    def initial_state(self, batch_size: int, **kwargs) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def add_measure(self,
                    measure: str,
                    state_element: str,
                    value: Union[float, Tensor, None]) -> None:

        assert state_element in self.state_elements, f"'{state_element}' is not in this process.'"

        key = (measure, state_element)

        if key in self.state_elements_to_measures.keys():
            raise ValueError(f"The (measure, state_element) '{key}' was already added to this process.")
        self.state_elements_to_measures[key] = value

    def parameters(self) -> Generator[Parameter, None, None]:
        raise NotImplementedError

    def covariance(self) -> Covariance:
        raise NotImplementedError

    @property
    def state_element_idx(self) -> Dict[str, int]:
        if self._state_element_idx is None:
            self._state_element_idx = {el: i for i, el in enumerate(self.state_elements)}
        return self._state_element_idx

    def for_batch(self, batch_size: int, **kwargs) -> 'ProcessForBatch':
        return ProcessForBatch(process=self, batch_size=batch_size)

    def set_to_simulation_mode(self, *args, **kwargs):
        """
        Set initial parameters to reasonable values s.t. generating data from this process in a simulation will be reasonable
        """
        self.requires_grad_(False)
