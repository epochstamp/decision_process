

from ..decision_process_components.parameter import Parameter
from ..decision_process_components.variable import Variable
from ..decision_process_components.expression.index_set import\
    Index, IndexSet
from typing import Dict, Tuple, Union, List, Callable
from ..decision_process_components.evaluator import Evaluator


class GymEvaluator(Evaluator):

    def __init__(self,
                 param_table: Dict[str,
                                   Union[Union[float, int],
                                         Dict[Tuple[Index, ...],
                                              Union[float, int]]]],
                 state_table: Dict[str,
                                   Union[Union[float, int],
                                         Dict[Tuple[Index, ...],
                                              Union[float, int]]]],
                 action_table: Dict[str,
                                    Union[Union[float, int],
                                          Dict[Tuple[Index, ...],
                                               Union[float, int]]]],
                 index_set_components_func: Callable[[IndexSet],
                                                     List[Index]],
                 index_table: Dict[IndexSet, Index]):
        self._param_table = param_table
        self._state_table = state_table
        self._action_table = action_table
        self._var_table = {**state_table, **action_table}
        self._index_set_components_func = index_set_components_func
        self._index_table = index_table

    def _get(self,
             container: Union[Variable, Parameter],
             idx_seq: List[Index]):
        if (isinstance(container, Parameter)):
            table = self._param_table
        elif isinstance(container, Variable):
            table = self._var_table

        return (table[container.id]
                if container.shape == tuple()
                else table[container.id][idx_seq])

    def _get_all_components_by_index(self, index_set: IndexSet) -> List[Index]:
        return self._index_set_components_func(index_set)

    def _get_index(self, index_set: IndexSet) -> Index:
        return self._index_table[index_set]
