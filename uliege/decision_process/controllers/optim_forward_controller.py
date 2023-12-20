# -*- coding: UTF-8 -*-
from ..datastream import DatastreamFactory, Datastream
from ..decision_process import Index, ParameterData, VariableData, DecisionProcess
from ..controllers import TrajectoryOptimizer
from abc import ABC, abstractmethod
from ..decision_process_components import IndexSet
from typing import Dict, Union, Tuple, List
from .forward_controller import ForwardController
from copy import deepcopy
import numpy as np
import pyomo.environ as pyo
import sys
from ..decision_process_components.cost_function import DiscountFactor


class DatastreamOverrider(Datastream):
    """
        Overrides the initial state carried
        of a Datastream object

        Attributes
        ----------
        datastream: Datastream
            A datastream object
        initial_state: dict of str to float/int/
                       dict of tuple of Index to float/int
            A given initial state identified by their ids
            and possibly indexed
    """

    def __init__(self,
                 datastream: Datastream,
                 initial_state: Dict[str,
                                     Union[float,
                                           int,
                                           Dict[Tuple[Index, ...],
                                                Union[float,
                                                      int]]]]):
        self._datastream = datastream
        self._initial_state = initial_state

    def _get_parameter(self,
                       param_data: ParameterData,
                       idx_seq: List[Index],
                       length: int):
        """
            Get the (possibly indexed) parameter from the overrided datastream
        """
        return self._datastream.get_parameter(param_data, idx_seq, length)

    def _get_indexes_by_index_set(self,
                                  index_set: IndexSet,
                                  indexes_source: List[Tuple[IndexSet, Index]] = []) -> Union[None,
                                                                List[Index]]:
        """
            Get the (possibly empty) set of indexes of an index set
            from the datastream
        """
        return self._datastream.get_indexes_by_index_set(index_set, indexes_source)

    def _get_initialization(self,
                            var_data: VariableData,
                            idx_seq: List[Index]):
        """
            Get the overrided initial state
        """
        if len(idx_seq) == 0:
            value = self._initial_state[var_data.id]
        else:
            idx_seq = tuple(idx_seq)
            value = self._initial_state[var_data.id][idx_seq]
        return value[-1] if isinstance(value, list) else value

    def activate_helper(self, var_data: VariableData, T: int) -> List[bool]:
        """
            Returns a boolean vector of length T which indicates whether a helper variable should be created each discrete time step between 0 and T-1
        """
        return self._datastream.activate_helper(var_data, T)


class OptimForwardController(ForwardController):
    """Wrapper for look-ahead optimization controller

       Parameters
       -----------
       optim_forward_controller: OptimForwardController
           An OptimForwardController object
       time_horizon: int (optional, default=1)
           Time horizon of the controller
           (useful for look-ahead during simulation)
           To be left as default if time horizon
           is not expected to change towards end of the simulation
       solver_verbose: bool (optional, default=False)
           Whether the solver trace should be displayed
       solver_factory: SolverFactory (optional, default=None)
           The object which specifies the solver to use
           (if None, automatically determined by the controller)
    """

    def __init__(self,
                 time_horizon: int = 1,
                 solver_verbose: bool = False,
                 solver_factory: Union[pyo.SolverFactory,
                                       type(None)] = None,
                 discount_factor: float = 1.0,
                 decision_process: DecisionProcess = None,
                 save_action_sequences: bool = False,
                 ):
        self._time_horizon = time_horizon
        self._current_time_horizon = time_horizon
        self._solver_verbose = solver_verbose
        self._solver_factory = solver_factory
        self._initialized = False
        self._datastream = None
        self._decision_process = decision_process
        self._discount_factor = discount_factor
        self._save_action_sequences = save_action_sequences

    def initialize(self,
                   decision_process: DecisionProcess,
                   datastream: Datastream,
                   control_time_horizon: int = 1,
                   ) -> None:
        """ Build and store a TrajectoryOptimizer object

            Parameters
            ----------
            decision_process: DecisionProcess
                A decision process object
            datastream: Datastream
                A datastream object (useful for lookahead models)
            control_time_horizon: int
                A strictly positive integer for control time horizon
        """
        decision_process =\
            self._decision_process if self._decision_process is not None\
                else decision_process
        self._trajectory_optimizer =\
            TrajectoryOptimizer(
                decision_process=decision_process,
                datastream=datastream,
                time_horizon=control_time_horizon,
                discount_factor=DiscountFactor(self._discount_factor)
            )
        self._action_sequences = []
        self._datastream = datastream
        self._initialized = True

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
        self._trajectory_optimizer.datastream = datastream
        self._trajectory_optimizer.time_horizon = control_time_horizon
        self._datastream = datastream
        self._time_horizon = control_time_horizon

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
        """ Returns the first action of an optimized look-ahead trajectory
            of decision process contexts.
            The size of the trajectory is the minimum between time horizon left
            and controller time horizon.

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
        current_datastream = self._datastream
        if state is not None:
            current_datastream = DatastreamOverrider(self._datastream, state)
        self._trajectory_optimizer.datastream = current_datastream
        self._trajectory_optimizer.time_horizon = self._time_horizon
        result = self._trajectory_optimizer.solve(
            solver_factory=self._solver_factory,
            solver_verbose=self._solver_verbose
        )

        if self._save_action_sequences:
            self._action_sequences.append(deepcopy(result.action_sequence))

        action = dict()
        for action_variable, sequence in result.action_sequence.items():
            if isinstance(sequence, list):
                if not sequence == []:
                    action[action_variable] = sequence[0]
            elif isinstance(sequence, dict):
                action[action_variable] = dict()
                for idx_seq, values in sequence.items():
                    action[action_variable][idx_seq] = values[0]
        return action

    @property
    def action_sequences(self) -> List:
        """ List of action sequences of the optimisations performed by the
            controller.
        """
        if not self._save_action_sequences:
            raise RuntimeError(
                'This controller is configured for NOT saving action sequences.'
            )

        return self._action_sequences
