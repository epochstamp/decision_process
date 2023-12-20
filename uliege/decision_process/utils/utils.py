from __future__ import annotations
from typing import Type, Union, Any, Dict, List, FrozenSet, Tuple

from ..decision_process_components.expression.numeric_expression import NumericExpression, NumericExpressionData


from ..decision_process_components.expression.expression import\
    Expression, ExpressionData
from ..decision_process_components.expression.index_set import Index, IndexSet
from ..decision_process_components.expression.indexed_container import\
    IndexedContainer, IndexedContainerData, IndexedVariableData,\
    IndexedParameterData
from ..decision_process_components.variable import Variable, VariableData
from ..decision_process_components.parameter import Parameter, ParameterData
from ..decision_process_components.expression.binop import Binop, BinopData, Binoperator
from ..decision_process_components.expression.unop import Unop, UnopData
from ..decision_process_components.expression.reduce import Reduce, ReduceData
from ..decision_process_components.expression.ineq import Ineq, IneqData
from typing import Hashable
from sortedcontainers import SortedKeyList
from bisect import bisect_left, bisect_right
import itertools
import numpy as np
from collections import defaultdict
import warnings


class SortedAndEqualedSetByKey(SortedKeyList):  # pragma: no cover

    def __contains__(self, value):
        """Return true if `value` is an element of the sorted-key list.

        ``skl.__contains__(value)`` <==> ``value in skl``

        Runtime complexity: `O(log(n))`

        >>> from operator import neg
        >>> skl = SortedKeyList([1, 2, 3, 4, 5], key=neg)
        >>> 3 in skl
        True

        :param value: search for value in sorted-key list
        :return: true if `value` in sorted-key list

        """
        _maxes = self._maxes

        if not _maxes:
            return False

        key = self._key(value)
        pos = bisect_left(_maxes, key)

        if pos == len(_maxes):
            return False

        _lists = self._lists
        _keys = self._keys

        idx = bisect_left(_keys[pos], key)

        len_keys = len(_keys)
        len_sublist = len(_keys[pos])

        while True:
            if _keys[pos][idx] != key:
                return False
            if self._key(_lists[pos][idx]) == key:
                return True
            idx += 1
            if idx == len_sublist:
                pos += 1
                if pos == len_keys:
                    return False
                len_sublist = len(_keys[pos])
                idx = 0

    def add(self, value):
        """Add `value` to sorted-key list.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> from operator import neg
        >>> skl = SortedKeyList(key=neg)
        >>> skl.add(3)
        >>> skl.add(1)
        >>> skl.add(2)
        >>> skl
        SortedKeyList([3, 2, 1], key=<built-in function neg>)

        :param value: value to add to sorted-key list

        """
        if value in self:
            return
        _lists = self._lists
        _keys = self._keys
        _maxes = self._maxes

        key = self._key(value)

        if _maxes:
            pos = bisect_right(_maxes, key)

            if pos == len(_maxes):
                pos -= 1
                _lists[pos].append(value)
                _keys[pos].append(key)
                _maxes[pos] = key
            else:
                idx = bisect_right(_keys[pos], key)
                _lists[pos].insert(idx, value)
                _keys[pos].insert(idx, key)

            self._expand(pos)
        else:
            _lists.append([value])
            _keys.append([key])
            _maxes.append(key)

        self._len += 1


def localize_dict_by_key(d: Dict,
                         k: Hashable,
                         height: int = -1) ->\
    Union[Dict,
          Tuple[Dict, int],
          type(None),
          Tuple[type(None), type(None)]]:
    """
        Localise a (nested) sub-dictionary by key

        Parameters
        ----------
        d : dict
            A dictionary
        k : hashable
            Any hashable object
        height: int
            A starting height. Value of -1 neutralizes it.

        Returns
        ----------
        dict or tuple of dict int, int or none or tuple of none, none
            A dictionary with only the key k
            and its corresponding subdict,
            or a tuple of this dictionary and a positive integer
            if height was set to a value > -1.
            (returns none or none, none if not found)
    """
    if not isinstance(d, dict):
        return None if height == -1 else (None, None)
    for key in d.keys():
        if k == key:
            return {k: d[key]} if height == -1 else ({k: d[key]}, height)
        else:
            new_height = -1 if height == -1 else (height + 1)
            dnest = localize_dict_by_key(d[key], k, height=new_height)
            if dnest is not None and dnest != (None, None):
                return dnest
    return None if height == -1 else (None, None)

def list_duplicates(seq):
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items() 
                            if len(locs)>1)


def extract_all_dict_leaves(d: Union[Dict, Any]) -> List[Any]:
    """
        Return all 'leaves' of the dictionary d
        (i.e., nested values of the dictionary that
         are not dictionaries)

        Parameters
        ----------
        d : dict
            A dictionary

        Returns
        ----------
        list
            A (possibly nested) list of objets that are not dicts
    """
    if not isinstance(d, dict):
        return [d]
    lst = []
    for key in d.keys():
        lst.extend(extract_all_dict_leaves(d[key]))
    return lst


def flatten(lst: list) -> list:
    return [item for sublist in lst for item in sublist]


def has_parent(index_set: IndexSet, index_set_parent: IndexSet) -> bool:
    """
        Test whether index_set inherits from index_set_parent

        Parameters
        ----------
        index_set: IndexSet
            An index set
        index_set_parent: IndexSet
            An index set

        Returns
        ----------
        bool
            True if index_set_parent == index_set
                 or index_set inherits from index_set_parent
    """
    if index_set == index_set_parent or index_set.parent == index_set_parent:
        return True
    if index_set.parent is None:
        return False
    else:
        return has_parent(index_set.parent, index_set_parent)


def convert_expr_to_indexed_container(expr: ExpressionData)\
        -> IndexedContainerData:
    """Explicitly cast an expression DTO to an
       indexed container DTO if possible

       Parameters
       ----------

       expr: ExpressionData
            The expression DTO to cast explicitly into IndexedContainer DTO

       Returns
       ----------
       IndexedContainer
            The expression casted to an IndexedContainer DTO

    """
    if isinstance(expr, IndexedContainerData):
        return expr
    else:
        error = "expr is not an IndexedContainerData DTO but a "\
                + expr.__class__.__name__ + " object"
        raise TypeError(error)


def convert_expr_to_indexed_variable(expr: ExpressionData)\
        -> IndexedVariableData:
    """Explicitly cast an expression DTO to an
       indexed variable DTO if possible

       Parameters
       ----------

       expr: ExpressionData
            The expression DTO to cast explicitly into IndexedVariable DTO

       Returns
       ----------
       IndexedVariable
            The expression casted to an IndexedVariable DTO

    """
    if isinstance(expr, IndexedVariableData):
        return expr
    else:
        error = "expr is not an IndexedVariableData DTO but a "\
                + expr.__class__.__name__ + " object"
        raise TypeError(error)


def convert_expr_to_indexed_parameter(expr: ExpressionData)\
        -> IndexedParameterData:
    """Explicitly cast an expression DTO to an
       indexed parameter DTO if possible

       Parameters
       ----------

       expr: ExpressionData
            The expression DTO to cast explicitly into IndexedParameter DTO

       Returns
       ----------
       IndexedParameter
            The expression casted to an IndexedParameter DTO

    """
    if isinstance(expr, IndexedParameterData):
        return expr
    else:
        error = "expr is not an IndexedParameterData DTO but a "\
                + expr.__class__.__name__ + " object"
        raise TypeError(error)


def convert_expr_to_binop(expr: ExpressionData) -> BinopData:
    """Explicitly cast an expression to a binary operation if possible

       Parameters
       ----------

       expr: ExpressionData
            The expression to cast explicitly into Binop DTO

       Returns
       ----------
       Binop
            The expression casted to a Binop DTO

    """
    if isinstance(expr, BinopData):
        return expr
    else:
        error = "expr is not a BinopData DTO but a "\
                + expr.__class__.__name__ + " object"
        raise TypeError(error)


def convert_expr_to_unop(expr: ExpressionData) -> UnopData:
    """Explicitly cast an expression to an unary operation if possible

       Parameters
       ----------

       expr: ExpressionData
            The expression to cast explicitly into Unop DTO

       Returns
       ----------
       Unop
            The expression casted to a Unop DTO

    """
    if isinstance(expr, UnopData):
        return expr
    else:
        error = "expr is not an UnopData expression but a "\
                + expr.__class__.__name__ + " object"
        raise TypeError(error)


def convert_expr_to_reduce(expr: ExpressionData) -> ReduceData:
    """Explicitly cast an expression to a reduce operation if possible

       Parameters
       ----------

       expr: ExpressionData
            The expression to cast explicitly into Reduce DTO

       Returns
       ----------
       Reduce
            The expression casted to a Reduce DTO

    """
    if isinstance(expr, ReduceData):
        return expr
    else:
        error = "expr is not an ReduceData expression but a "\
                + expr.__class__.__name__ + " object"
        raise TypeError(error)


def convert_expr_to_ineq(expr: ExpressionData) -> IneqData:
    """Explicitly cast an expression to an inequation if possible

       Parameters
       ----------

       expr: ExpressionData
            The expression to cast explicitly into Ineq DTO

       Returns
       ----------
       Reduce
            The expression casted to a Ineq DTO

    """
    if isinstance(expr, IneqData):
        return expr
    else:
        error = "expr is not an Inequation expression but a "\
                + expr.__class__.__name__ + " object"
        raise TypeError(error)


def reduce_index_sets(expr: ExpressionData) -> FrozenSet[IndexSet]:
    """Build the set of all index sets that are involved
       in (nested) Reduce expressions

       Parameters
       ----------
       expr : ExpressionData
            The expression to dive in for free index sets

       Returns
       ----------
       set of indexset
            The set of free index sets inside the expression

    """
    if isinstance(expr, IndexedContainerData):
        return set()
    elif isinstance(expr, BinopData):
        binop = convert_expr_to_binop(expr)
        expr_1 = binop.expr_1
        expr_2 = binop.expr_2
        return reduce_index_sets(expr_1).union(reduce_index_sets(expr_2))
    elif isinstance(expr, UnopData):
        unop = convert_expr_to_unop(expr)
        expr = unop.expr
        return reduce_index_sets(expr)
    elif isinstance(expr, ReduceData):
        reducer = convert_expr_to_reduce(expr)
        expr = reducer.inner_expr
        reduce_index_expr = reduce_index_sets(expr)
        idx_sets = reduce_index_expr.union({reducer.idx_reduce_set.idx_set})
        return idx_sets
    elif isinstance(expr, IneqData):
        inequation = convert_expr_to_ineq(expr)
        expr_1 = inequation.expr_1
        expr_2 = inequation.expr_2
        return reduce_index_sets(expr_1).union(reduce_index_sets(expr_2))
    else:
        raise NotImplementedError(expr.__class__.__name__)


def free_index_sets(expr: ExpressionData,
                    attached: bool = False,
                    keep_partial_free_if_attached = False) -> FrozenSet[IndexSet]:
    """Build the set of all index sets inside expressions
       except of index sets referred by ReduceIndexSet objects

       Parameters
       ----------
       expr : ExpressionData
            The expression to dive in for free index sets
       attached : bool
            Whether the index sets in containers should be returned separately
       keep_partial_free_if_attached : bool
            If `attached`, whether to return concatenated index sets even if some of them are part of a reduce expression

       Returns
       ----------
       set of indexset
            The set of free index sets inside the expression

    """
    if isinstance(expr, IndexedContainerData):
        container = convert_expr_to_indexed_container(expr)
        idx_seq = container.idx_seq
        if idx_seq == tuple(): 
            return set()
        elif isinstance(idx_seq, IndexSet):
            return set((idx_seq,))
        return set([idx_seq]) if attached else set(idx_seq)
    elif isinstance(expr, BinopData):
        binop = convert_expr_to_binop(expr)
        expr_1 = binop.expr_1
        expr_2 = binop.expr_2
        return free_index_sets(
            expr_1, attached, keep_partial_free_if_attached
        ).union(
            free_index_sets(expr_2, attached, keep_partial_free_if_attached)
        )
    elif isinstance(expr, UnopData):
        unop = convert_expr_to_unop(expr)
        expr = unop.expr
        return free_index_sets(expr, attached, keep_partial_free_if_attached)
    elif isinstance(expr, ReduceData):
        reducer = convert_expr_to_reduce(expr)
        expr = reducer.inner_expr
        idx_set = reducer.idx_reduce_set.idx_set
        free_idx_sets = free_index_sets(expr, attached, keep_partial_free_if_attached)
        if not attached or keep_partial_free_if_attached:
            return free_idx_sets.difference({idx_set})
        else:
            return {idx_seq for idx_seq in free_idx_sets if idx_set not in idx_seq}
    elif isinstance(expr, IneqData):
        inequation = convert_expr_to_ineq(expr)
        expr_1 = inequation.expr_1
        expr_2 = inequation.expr_2
        return free_index_sets(expr_1, attached).union(free_index_sets(expr_2, attached))
    else:
        raise NotImplementedError(expr.__class__.__name__)


def index_sets(expr: ExpressionData, attached: bool = False) -> FrozenSet[IndexSet]:
    """Build the set of all index sets that are involved
       in expressions (regardless of being free or involved in a Reduce expression)

       Parameters
       ----------
       expr : ExpressionData
            The expression to dive in for index sets
        attached : bool
            Whether the index sets in containers should be returned separately

       Returns
       ----------
       set of indexset
            The set of index sets inside the expression

    """
    return free_index_sets(expr, attached).union(reduce_index_sets(expr))


def get_variables(expr: ExpressionData,
                  true_objects: bool = False) -> FrozenSet[Union[Variable,
                                                                 str]]:
    """Build the set of all variables inside the expression

       Parameters
       ----------
       expr : ExpressionData
            The expression to dive in for variables
       true_objects : bool
            Whether the ids or the objects themselves
            are returned

       Returns
       ----------
       set of str or set of Variable
            The set of variables inside the expression

    """
    if isinstance(expr, IndexedContainerData):
        indexed_container = convert_expr_to_indexed_container(expr)
        container = indexed_container.container
        if isinstance(container, VariableData):
            return ({container.id}
                    if not true_objects
                    else {Variable.from_data(container)})
        else:
            return set()
    elif isinstance(expr, BinopData):
        binop = convert_expr_to_binop(expr)
        expr_1 = binop.expr_1
        expr_2 = binop.expr_2
        return get_variables(expr_1, true_objects).union(get_variables(expr_2, true_objects))
    elif isinstance(expr, UnopData):
        expr = convert_expr_to_unop(expr).expr
        return get_variables(expr, true_objects)
    elif isinstance(expr, ReduceData):
        reducer = convert_expr_to_reduce(expr)
        expr = reducer.inner_expr
        idx_set = reducer.idx_reduce_set.idx_set
        return get_variables(expr, true_objects).difference({idx_set})
    elif isinstance(expr, IneqData):
        inequation = convert_expr_to_ineq(expr)
        expr_1 = inequation.expr_1
        expr_2 = inequation.expr_2
        return get_variables(expr_1, true_objects).union(get_variables(expr_2, true_objects))
    else:
        raise NotImplementedError(expr.__class__.__name__)


def get_parameters(expr: ExpressionData,
                   true_objects: bool = False) -> FrozenSet[Union[Parameter,
                                                                  str]]:
    """Build the set of all variable inside the expression

       Parameters
       ----------
       expr : ExpressionData
            The expression to dive in for variables
       true_objects : bool
            Whether the ids or the objects themselves
            are returned

       Returns
       ----------
       set of str or set of Parameter
            The set of variables inside the expression

    """
    if isinstance(expr, IndexedContainerData):
        indexed_container = convert_expr_to_indexed_container(expr)
        container = indexed_container.container
        if isinstance(container, ParameterData):
            return {container.id}
        else:
            return set()
    elif isinstance(expr, BinopData):
        binop = convert_expr_to_binop(expr)
        expr_1 = binop.expr_1
        expr_2 = binop.expr_2
        return get_parameters(expr_1).union(get_parameters(expr_2))
    elif isinstance(expr, UnopData):
        unop = convert_expr_to_unop(expr)
        expr = unop.expr
        return get_parameters(expr)
    elif isinstance(expr, ReduceData):
        reducer = convert_expr_to_reduce(expr)
        expr = reducer.inner_expr
        return get_parameters(expr)
    elif isinstance(expr, IneqData):
        inequation = convert_expr_to_ineq(expr)
        expr_1 = inequation.expr_1
        expr_2 = inequation.expr_2
        return get_parameters(expr_1).union(get_parameters(expr_2))
    else:
        raise NotImplementedError(expr.__class__.__name__)


def get_expr_from_data(expr_data: ExpressionData) -> Expression:
    """
        Returns an expression object from its respective DTO

        Parameters
        ----------
        expr_data: ExpressionData
            Any inherited object from ExpressionData

        Returns
        ----------
        Expression
            The expression object derived from its respective DTO

        Raises
        ----------
        NotImplementedError

            If the expression is not handled by the function
            (abstract or unhandled yet objects)
    """
    expr_cls = Expression
    if isinstance(expr_data, IndexedContainerData):
        if isinstance(expr_data, IndexedVariableData):
            expr_data = convert_expr_to_indexed_variable(expr_data)
        elif isinstance(expr_data, IndexedParameterData):
            expr_data = convert_expr_to_indexed_parameter(expr_data)
        expr_cls = IndexedContainer
    elif isinstance(expr_data, BinopData):
        expr_cls = Binop
    elif isinstance(expr_data, UnopData):
        expr_cls = Unop
    elif isinstance(expr_data, ReduceData):
        expr_cls = Reduce
    elif isinstance(expr_data, IneqData):
        expr_cls = Ineq
    else:
        raise NotImplementedError(expr_data.__class__.__name__)
    return expr_cls.from_data(expr_data)


class NewDecisionProcessInvalidAfterPreProcessing(BaseException):
    pass


def trim_expression_from_undefined_containers(numeric_expr_data: NumericExpressionData,
                                              container_set: FrozenSet[str]) -> Union[NumericExpression, Type[None]]:
    if isinstance(numeric_expr_data, IndexedContainerData):
        if numeric_expr_data.container.id not in container_set:
            return None
    elif isinstance(numeric_expr_data, BinopData):
        trimmed_expr_1 = trim_expression_from_undefined_containers(numeric_expr_data.expr_1, container_set)
        trimmed_expr_2 = trim_expression_from_undefined_containers(numeric_expr_data.expr_2, container_set)
        if (numeric_expr_data.binop == Binoperator.MUL or numeric_expr_data.binop == Binoperator.DIV) and (trimmed_expr_1 is None or trimmed_expr_2 is None):
            return None
        if trimmed_expr_1 is None:
            return trimmed_expr_2
        elif trimmed_expr_2 is None:
            return trimmed_expr_1
        else:
            return BinopData(expr_1=trimmed_expr_1, binop=numeric_expr_data.binop, expr_2=trimmed_expr_2)
    elif isinstance(numeric_expr_data, UnopData):
        trimmed_expr = trim_expression_from_undefined_containers(numeric_expr_data.expr, container_set)
        if trimmed_expr is None:
            return None
        else:
            return UnopData(expr=trimmed_expr, unop=numeric_expr_data.unop)
    elif isinstance(numeric_expr_data, ReduceData):
        trimmed_expr = trim_expression_from_undefined_containers(numeric_expr_data.inner_expr, container_set)
        if trimmed_expr is None:
            return None
        else:
            return ReduceData(inner_expr=trimmed_expr, idx_reduce_set=numeric_expr_data.idx_reduce_set)
    else:
        raise NotImplementedError(numeric_expr_data.__class__.__name__)

    return numeric_expr_data


def transform_decision_process_with_index_sets_mapping(
        complete_decision_process: DecisionProcess,
        index_set_mappings: Dict[Tuple[IndexSet, ...], List[Tuple[Index, ...]]]):
    """ Transforms a decision process to remove:
        - Any variable or parameter for which one of the shape
            leads to an empty index set
        - Any variable or parameter for which any combination
            of the indexes from the index sets does not make sense
            according to the datastream
        - Any function which make use of an indexed
            variable/parameter where one of the index sets
            is an empty one

        Parameters
        ----------
        complete_decision_process: DecisionProcess
            The complete decision process to transform
        datastream: Datastream
            The datastream used to transforms decision_process

        Returns
        ----------
        decision_process: DecisionProcess
            A new and validated DecisionProcess

        Raises
        ----------
        NewDecisionProcessInvalidAfterPreProcessing

            If the new decision process cannot be validated

    """
    from ..decision_process import (
        DecisionProcess,
        DecisionProcessError,
        NotDefinedVariableError,
        Constraint,
        NotDefinedParameterError,
        CostFunction,
        Dynamics
    )
    from ..datastream.datastream import Datastream
    decision_process = DecisionProcess(id=complete_decision_process.id)
    variable_adders = ((decision_process.add_state_variables,
                        complete_decision_process.state_variables),
                       (decision_process.add_action_variables,
                        complete_decision_process.action_variables),
                       (decision_process.add_helper_variables,
                        complete_decision_process.helper_variables))

    for variable_adder, variable_set in variable_adders:
        for variable in variable_set:
            vshape = variable.shape if\
                isinstance(variable.shape, tuple) else (variable.shape,)
            if vshape == tuple() or index_set_mappings[vshape] != []:
                variable_adder(variable)
    for parameter in complete_decision_process.parameters:
        pshape = parameter.shape if\
                isinstance(parameter.shape, tuple) else (parameter.shape,)
        if pshape == tuple() or index_set_mappings[pshape] != []:
            decision_process.add_parameters(parameter)

    containers = {v.id for v in decision_process.variables}.union({p.id for p in decision_process.parameters})

    for dynamics in complete_decision_process.dynamics_functions:
        all_index_sets_state_var = dynamics.state_var.get_data().idx_seq
        if all_index_sets_state_var == tuple() or index_set_mappings[all_index_sets_state_var] != []:
            try:
                decision_process.add_dynamics_functions(dynamics)
            except (NotDefinedVariableError, NotDefinedParameterError) as e:
                expr = trim_expression_from_undefined_containers(dynamics.state_var_update.get_data(), containers)
                if expr is not None:
                    new_dynamics = Dynamics(
                        dynamics.state_var,
                        get_expr_from_data(expr),
                        dynamics.id,
                        dynamics.description
                    )
                    decision_process.add_dynamics_functions(new_dynamics, containers)
                    warning = "Dynamic(s) left-hand-side expression {" + dynamics.id + "} had to be trimmed (subexpressions removed) due to"
                    warning += f"variable/parameter removing. Details : {e}"
                    warnings.warn(warning)
                else:
                    warnings.warn("Dynamics {"+dynamics.id+"} had to be skipped. Details: " + str(e))


    for constraint in complete_decision_process.constraint_functions:
        if constraint.free_idx_sets == [] or index_set_mappings[constraint.free_idx_sets[0]] != []:
            try:
                decision_process.add_constraint_functions(constraint)
            except (NotDefinedVariableError, NotDefinedParameterError) as e:
                
                ineq_data = constraint.ineq.get_data()
                expr_1 = trim_expression_from_undefined_containers(ineq_data.expr_1, containers)
                expr_2 = trim_expression_from_undefined_containers(ineq_data.expr_2, containers)
                if expr_1 is not None and expr_2 is not None:
                    ineq = Ineq.from_data(
                        IneqData(
                            expr_1=expr_1, ineq_op=ineq_data.ineq_op, expr_2=expr_2
                        )
                    )
                    new_constraint = Constraint(ineq,
                                                id=constraint.id,
                                                description=constraint.description,
                                                shift_time_state=constraint.shift_time_state)
                    decision_process.add_constraint_functions(new_constraint)
                    warning = "Constraint(s) {" + constraint.id + "} had to be trimmed (subexpressions removed) due to"
                    warning += f"variable/parameter removing. Details : {e}"
                    warnings.warn(warning)
                else:
                    warnings.warn("Constraint {"+constraint.id+"} had to be skipped. Details: " + str(e))

    for cost_function in complete_decision_process.cost_functions:
        all_index_sets = reduce_index_sets(cost_function.cost_expression.get_data())
        if all_index_sets == set():
            decision_process.add_cost_functions(cost_function)
            continue
        all_index_sets_empty = True
        for idx_set in all_index_sets:
            if index_set_mappings[(idx_set,)] != []:
                all_index_sets_empty = False
                break
        if not all_index_sets_empty:
            try:
                decision_process.add_cost_functions(cost_function)
            except (NotDefinedVariableError, NotDefinedParameterError) as e:
                expr = trim_expression_from_undefined_containers(cost_function.cost_expression.get_data(), containers)
                if expr is not None:
                    expr = get_expr_from_data(expr)
                    new_cost_function = CostFunction(
                        cost_expression=expr,
                        id=cost_function.id,
                        description=cost_function.description,
                        horizon_mask=cost_function.horizon_mask
                    )
                    decision_process.add_cost_functions(new_cost_function)
                    warning = "Cost function(s) {" + cost_function.id + "} had to be trimmed (subexpressions removed) due to"
                    warning += f"variable/parameter removing. Details : {e}"
                    warnings.warn(warning)
                else:
                    warnings.warn("Cost function {"+cost_function.id+"} had to be skipped. Details: " + str(e))

    try:
        decision_process.validate()
        return decision_process
    except DecisionProcessError as e:
        err = "The original decision process submitted "
        err += "has been validated but needed to be preprocessed "
        err += "to remove variables/parameters for which at least "
        err += "one of its index sets are empty according to the datastream "
        err += "and expressions that either used them or used indexed "
        err += "variables/parameters with empty index subsets.\n "
        err += "Unfortunately, the new decision process resulting "
        err += "from this preprocessing made the latter invalid.\n"
        err += "Details : " + str(e)
        raise NewDecisionProcessInvalidAfterPreProcessing(err)
