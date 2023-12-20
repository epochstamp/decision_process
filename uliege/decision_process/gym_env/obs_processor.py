from typing import Dict, Union, Tuple, List, Callable
from ..decision_process_components.expression.index_set import Index
from ..decision_process_components.variable import Variable, is_real
from ..decision_process_components.parameter import Parameter
from ..decision_process_components.expression.index_set import Index, IndexSet
from ..decision_process import DecisionProcess
import numpy as np
from gym.spaces import Space
import itertools
from ..datastream.datastream import Datastream
from ..gym_env.util import get_dict_space_from_container_set
from abc import ABC, abstractmethod


class ObsProcessor(ABC):

    @abstractmethod
    def get_space(self,
                  decision_process: DecisionProcess,
                  components_by_index_set_func: Callable[[IndexSet],
                                                         List[Index]])\
            -> Space:
        raise NotImplementedError()

    @abstractmethod
    def __call__(
        self,
        states: Dict[Variable, Union[Union[int, float],
                                     Dict[Tuple[Index, ...],
                                     Union[int, float]]]],
        parameters: Dict[Parameter, Union[Union[int, float],
                                          Dict[Tuple[Index, ...],
                                          Union[int, float]]]],
        datastream: Datastream)\
            -> np.array:
        raise NotImplementedError()


class StateSpace(ObsProcessor):

    def get_space(
        self,
        decision_process: DecisionProcess,
        components_by_index_set_func: Callable[[IndexSet],
                                               List[Index]])\
            -> Space:
        spaces = list(decision_process.state_variables) + \
                 list(decision_process.parameters)
        return get_dict_space_from_container_set(spaces,
                                                 components_by_index_set_func)

    def __call__(
        self,
        states: Dict[Variable, Union[Union[int, float],
                                     Dict[Tuple[Index, ...],
                                     Union[int, float]]]],
        parameters: Dict[Parameter, Union[Union[int, float],
                                          Dict[Tuple[Index, ...],
                                          Union[int, float]]]],
        datastream: Datastream
        )\
            -> np.array:
        return states
