from typing import Dict, Union, Tuple  # pragma: no cover
from ..decision_process_components.expression.index_set import\
    Index  # pragma: no cover
from ..decision_process_components.variable import Variable  # pragma: no cover
from ..decision_process_components.parameter import\
    Parameter  # pragma: no cover
from ..datastream.datastream import Datastream  # pragma: no cover
from abc import ABC, abstractmethod  # pragma: no cover


class Drawer(ABC):  # pragma: no cover

    @abstractmethod
    def draw(self,
             states: Dict[Variable, Union[Union[int, float],
                                          Dict[Tuple[Index, ...],
                                          Union[int, float]]]],
             parameters: Dict[Parameter, Union[Union[int, float],
                                               Dict[Tuple[Index, ...],
                                                    Union[int, float]]]],
             actions: Dict[Variable, Union[Union[int, float],
                                           Dict[Tuple[Index, ...],
                                                Union[int, float]]]],
             datastream: Datastream) -> None:
        raise NotImplementedError()

    @abstractmethod
    def close(self):
        raise NotImplementedError()


class NoDrawer(Drawer):  # pragma: no cover

    def draw(self,
             states: Dict[Variable, Union[Union[int, float],
                                          Dict[Tuple[Index, ...],
                                          Union[int, float]]]],
             parameters: Dict[Parameter, Union[Union[int, float],
                                               Dict[Tuple[Index, ...],
                                                    Union[int, float]]]],
             actions: Dict[Variable, Union[Union[int, float],
                                           Dict[Tuple[Index, ...],
                                                Union[int, float]]]],
             datastream: Datastream) -> None:
        pass

    def close(self):
        pass
