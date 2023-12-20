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


class ActProcessor(ABC):
    """
        Interface for objects able to
        provide a surrogate set of actions
        (useful when original actions break the constraints)
    """

    @abstractmethod
    def __call__(
        self,
        states: Dict[str, Union[Union[int, float],
                                Dict[Tuple[Index, ...],
                                Union[int, float]]]],
        parameters: Dict[str, Union[Union[int, float],
                                    Dict[Tuple[Index, ...],
                                    Union[int, float]]]],
        actions: Dict[str, Union[Union[int, float],
                                 Dict[Tuple[Index, ...],
                                 Union[int, float]]]],
        datastream: Datastream,
        helpers: Dict[str, Union[Union[int, float],
                                 Dict[Tuple[Index, ...],
                                 Union[int, float]]]] = dict())\
            -> Dict[str, Union[Union[int, float],
                               Dict[Tuple[Index, ...],
                               Union[int, float]]]]:
        """
            Postprocess the actions given the full state
            of the decision process simulation
        """
        raise NotImplementedError()


class ActionSpace(ActProcessor):
    """
        Identity processor
    """

    def __call__(
        self,
        states: Dict[str, Union[Union[int, float],
                                Dict[Tuple[Index, ...],
                                Union[int, float]]]],
        parameters: Dict[str, Union[Union[int, float],
                                    Dict[Tuple[Index, ...],
                                    Union[int, float]]]],
        actions: Dict[str, Union[Union[int, float],
                                 Dict[Tuple[Index, ...],
                                 Union[int, float]]]],
        datastream: Datastream)\
            -> Dict[str, Union[Union[int, float],
                               Dict[Tuple[Index, ...],
                               Union[int, float]]]]:
        """
            Returns the same action as provided in parameters
        """
        return actions
