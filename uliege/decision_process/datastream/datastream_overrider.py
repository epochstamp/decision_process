from .datastream import Datastream
from typing import Dict, Union, Tuple, List
from ..decision_process import Index, ParameterData, VariableData
from ..decision_process_components import IndexSet


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
                                                      int]]]],
                 parameters: Dict[str,
                                     Union[List[float],
                                           List[int],
                                           float,
                                           int,
                                           Dict[Tuple[Index, ...],
                                                Union[List[float],
                                                      List[int],
                                                      float,
                                                      int]]]] = None):
        self._datastream = datastream
        self._initial_state = initial_state
        self._parameters = parameters

    def _get_parameter(self,
                       param_data: ParameterData,
                       idx_seq: List[Index],
                       length: int):
        """
            Get the (possibly indexed) parameter from the overrided datastream
        """
        if self._parameters is not None:
            if param_data.id in self._parameters:
                if len(idx_seq) == 0:
                    if not isinstance(self._parameters[param_data.id], list):
                        return self._parameters[param_data.id]
                    return self._parameters[param_data.id][:length]
                else:
                    if tuple(idx_seq) in self._parameters[param_data.id]:
                        if not isinstance(self._parameters[param_data.id][tuple(idx_seq)], list):
                            return self._parameters[param_data.id][tuple(idx_seq)]
                        return self._parameters[param_data.id][tuple(idx_seq)][:length]
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
            if var_data.id in self._initial_state:
                value = self._initial_state[var_data.id]
            else: value = self._datastream.get_initialization(var_data, idx_seq)
        else:
            idx_seq = tuple(idx_seq)
            if var_data.id in self._initial_state and idx_seq in self._initial_state[var_data.id]:
                value = self._initial_state[var_data.id][idx_seq]
            else: value = self._datastream.get_initialization(var_data, idx_seq)
        return value[-1] if isinstance(value, list) else value

    def activate_helper(self, var_data: VariableData, T: int) -> List[bool]:
        """
            Returns a boolean vector of length T which indicates whether a helper variable should be created each discrete time step between 0 and T-1
        """
        return self._datastream.activate_helper(var_data, T)
