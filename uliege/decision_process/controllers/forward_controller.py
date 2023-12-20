# -*- coding: UTF-8 -*-

from ..decision_process import DecisionProcessRealisation, DecisionProcess
from ..datastream import Datastream
from abc import ABC, abstractmethod
from ..decision_process_components import IndexSet
from typing import Dict, Union, Tuple


class ForwardController(ABC):
    """Forward controller interface
    """

    @abstractmethod
    def action(self,
               state: Dict[str, Union[int,
                                      float,
                                      Dict[Tuple[IndexSet,
                                                 ...],
                                           Union[int,
                                                 float]]]] = None)\
            -> Dict[str, Union[int,
                               float,
                               Dict[Tuple[IndexSet,
                                          ...],
                                    Union[int,
                                          float]]]]:
        """ Returns an action using a dict.
            Must be overrided by inherited classes.

            Parameters
            ----------
            state: dict of str and either a float/int value
                   or a dict of index tuple and float/int value
                The state of the decision process. If None (default), initial state
                will be the one defined by current datastream

            Returns
            ----------
            dict of str and either a float/int value
            or a dict of index tuple and float/int value
                A dict which provides action values
        """
        raise NotImplementedError()

    @abstractmethod
    def initialize(self,
                   decision_process: DecisionProcess,
                   datastream: Datastream,
                   control_time_horizon: int = 1) -> None:
        """ Initialize the controller

            Parameters
            ----------
            decision_process: DecisionProcess
                A decision process object
            datastream: Datastream
                A datastream object (useful for lookahead models)
            control_time_horizon: int
                A strictly positive integer for control time horizon
        """
        raise NotImplementedError()

    @abstractmethod
    def update(self,
               datastream: Datastream,
               control_time_horizon: int = 1) -> None:
        """ Notify a new decision process context for the controller

            Parameters
            ----------
            datastream: Datastream
                A datastream object (useful for lookahead models)
            control_time_horizon: int
                A strictly positive integer for control time horizon
        """
        raise NotImplementedError()
