import pytest
from uliege.decision_process.decision_process_components.expression.indexed_container import\
    IndexedContainer, IndexedContainerData
from uliege.decision_process.decision_process_components.expression.index_set import\
                                                 IndexSet
from uliege.decision_process.decision_process_components.expression.binop import Binop,\
                                                         BinopData
from uliege.decision_process.decision_process_components.expression.unop import Unop,\
                                                        UnopData
from uliege.decision_process.decision_process_components.expression.reduce import Reduce,\
                                                          ReduceData
from uliege.decision_process.decision_process_components.expression.expression import\
    ExpressionData
from uliege.decision_process.decision_process_components.expression.numeric_expression import\
    NumericExpression
from uliege.decision_process.decision_process_components.expression.ineq import\
    IneqData, Ineqoperator
from hypothesis import given, assume, settings, HealthCheck
from hypothesis.strategies import builds,\
                                  one_of,\
                                  lists,\
                                  recursive,\
                                  sampled_from
import random
import numpy as np
from typing import Union, Tuple, List, Dict, Hashable, FrozenSet
from .utils import reasonable_lenghty_text,\
                        build_unoperator_strat,\
                        build_binoperator_strat,\
                        build_reduce_strat,\
                        build_indexed_parameter,\
                        build_indexed_variable,\
                        dict_with_maps,\
                        nested_dict_with_maps,\
                        give_birth_to_index_set,\
                        build_indexed_parameter_detailed,\
                        build_indexed_variable_detailed,\
                        build_binoperator_strat_detailed,\
                        build_unoperator_strat_detailed,\
                        build_reduce_strat_detailed,\
                        flatten
from uliege.decision_process.utils.utils import localize_dict_by_key,\
                        extract_all_dict_leaves,\
                        has_parent,\
                        convert_expr_to_indexed_container,\
                        convert_expr_to_binop,\
                        convert_expr_to_unop,\
                        convert_expr_to_reduce,\
                        convert_expr_to_ineq,\
                        free_index_sets,\
                        reduce_index_sets,\
                        get_variables,\
                        get_parameters,\
                        get_expr_from_data,\
                        convert_expr_to_indexed_parameter,\
                        convert_expr_to_indexed_variable
import uuid

np.random.seed(1000)
random.seed(1000)


@given(recursive(base=dict_with_maps(),
                 extend=lambda s: nested_dict_with_maps(s)()))
def test_localize_dict_by_key(dwk: Tuple[List[Tuple[Hashable,
                                                    Union[Dict,
                                                          List[str]]]], Dict]):
    maps, nested_dict = dwk
    k, d = maps[np.random.choice(range(len(maps)))]
    assert(localize_dict_by_key(nested_dict, k) == {k: d})


@given(recursive(base=dict_with_maps(),
                 extend=lambda s: nested_dict_with_maps(s)()))
def test_localize_dict_by_notin_key_fail(dwk: Tuple[List[Tuple[
                                                         Hashable,
                                                         Union[Dict,
                                                               List[str]]]],
                                                    Dict]):
    _, nested_dict = dwk
    assert(localize_dict_by_key(nested_dict, uuid.uuid4()) is None)

@given(recursive(base=dict_with_maps(),
                 extend=lambda s: nested_dict_with_maps(s)()))
def test_all_dict_leaves(dwk: Tuple[List[Tuple[
                                         Hashable,
                                         Union[Dict,
                                               List[str]]]],
                                    Dict]):
    maps, nested_dict = dwk
    leaves = [v for _, v in maps if not isinstance(v, Dict)]
    assert(leaves == extract_all_dict_leaves(nested_dict))


@given(reasonable_lenghty_text)
def test_indexset_parent_of_itself(id: str):
    assert(has_parent(IndexSet(id), IndexSet(id)))


@given(lists(reasonable_lenghty_text,
             min_size=2,
             max_size=4))
def test_indexset_root_ancestor_is_parent(lst_ids: List[str]):
    previous = None
    lst_idxs = list()
    for id in lst_ids:
        index_set = IndexSet(id=id, parent=previous)
        lst_idxs.append(index_set)
        previous = index_set
    assert(has_parent(lst_idxs[-1], lst_idxs[0]))


@settings(suppress_health_check=(HealthCheck.too_slow,))
@given(recursive(base=builds(IndexSet,
                             id=reasonable_lenghty_text),
                 extend=lambda s: lists(give_birth_to_index_set(s)(),
                                        min_size=3),  # Birth control is out
                 max_leaves=8))  # Although...
def test_indexset_mother_of_all(idx_set: Union[IndexSet, List[IndexSet]]):
    if isinstance(idx_set, list):
        idx_set_len = len(idx_set)
        idx_set = flatten(idx_set)
        idx_set = idx_set[np.random.choice(range(idx_set_len))]
    parent = idx_set.parent
    if parent is not None:
        while parent.parent is not None:
            parent = parent.parent
    else:
        parent = idx_set
    assert(has_parent(idx_set, parent))


@given(one_of(build_indexed_parameter(),
              build_indexed_variable()))
def test_convert_indexed_container_identity(idx_container: IndexedContainer):
    expr_convert = convert_expr_to_indexed_container(idx_container.get_data())
    assert(isinstance(expr_convert, IndexedContainerData))


@given(recursive(base=one_of(build_indexed_parameter(),
                             build_indexed_variable()),
                 extend=lambda s: one_of(build_binoperator_strat(s)(),
                                         build_unoperator_strat(s)(),
                                         build_reduce_strat(s)()),
                 max_leaves=4))
def test_convert_expr_to_indexed_container_fail(expr: NumericExpression):
    assume(not isinstance(expr, IndexedContainer))
    with pytest.raises(TypeError):
        convert_expr_to_indexed_container(expr.get_data())


@given(build_indexed_parameter())
def test_convert_expr_to_indexed_variable_fail(
        idx_container: IndexedContainer):
    expr_convert = convert_expr_to_indexed_container(idx_container.get_data())
    with pytest.raises(TypeError):
        convert_expr_to_indexed_variable(expr_convert)


@given(build_indexed_variable())
def test_convert_expr_to_indexed_parameter_fail(
        idx_container: IndexedContainer):
    expr_convert = convert_expr_to_indexed_container(idx_container.get_data())
    with pytest.raises(TypeError):
        convert_expr_to_indexed_parameter(expr_convert)


@given(recursive(base=one_of(build_indexed_parameter(),
                             build_indexed_variable()),
                 extend=lambda s: one_of(build_binoperator_strat(s)(),
                                         build_unoperator_strat(s)(),
                                         build_reduce_strat(s)()),
                 max_leaves=8))
def test_convert_binop_identity(expr: NumericExpression):
    assume(isinstance(expr, Binop))
    expr_convert = convert_expr_to_binop(expr.get_data())
    assert(isinstance(expr_convert, BinopData))


@given(recursive(base=one_of(build_indexed_parameter(),
                             build_indexed_variable()),
                 extend=lambda s: one_of(build_binoperator_strat(s)(),
                                         build_unoperator_strat(s)(),
                                         build_reduce_strat(s)()),
                 max_leaves=8))
def test_convert_expr_to_binop_fail(expr: NumericExpression):
    assume(not isinstance(expr, Binop))
    with pytest.raises(TypeError):
        convert_expr_to_binop(expr.get_data())


@given(recursive(base=one_of(build_indexed_parameter(),
                             build_indexed_variable()),
                 extend=lambda s: one_of(build_binoperator_strat(s)(),
                                         build_unoperator_strat(s)(),
                                         build_reduce_strat(s)()),
                 max_leaves=8))
def test_convert_unop_identity(expr: NumericExpression):
    assume(isinstance(expr, Unop))
    expr_convert = convert_expr_to_unop(expr.get_data())
    assert(isinstance(expr_convert, UnopData))


@given(recursive(base=one_of(build_indexed_parameter(),
                             build_indexed_variable()),
                 extend=lambda s: one_of(build_binoperator_strat(s)(),
                                         build_unoperator_strat(s)(),
                                         build_reduce_strat(s)()),
                 max_leaves=8))
def test_convert_expr_to_unop_fail(expr: NumericExpression):
    assume(not isinstance(expr, Unop))
    with pytest.raises(TypeError):
        convert_expr_to_unop(expr.get_data())


@given(recursive(base=one_of(build_indexed_parameter(),
                             build_indexed_variable()),
                 extend=lambda s: one_of(build_binoperator_strat(s)(),
                                         build_unoperator_strat(s)(),
                                         build_reduce_strat(s)()),
                 max_leaves=8))
def test_convert_reduce_identity(expr: NumericExpression):
    assume(isinstance(expr, Reduce))
    expr_convert = convert_expr_to_reduce(expr.get_data())
    assert(isinstance(expr_convert, ReduceData))


@given(recursive(base=one_of(build_indexed_parameter(),
                             build_indexed_variable()),
                 extend=lambda s: one_of(build_binoperator_strat(s)(),
                                         build_unoperator_strat(s)(),
                                         build_reduce_strat(s)()),
                 max_leaves=8))
def test_convert_expr_to_reduce_fail(expr: NumericExpression):
    assume(not isinstance(expr, Reduce))
    with pytest.raises(TypeError):
        convert_expr_to_reduce(expr.get_data())


@given(recursive(base=one_of(build_indexed_parameter(),
                             build_indexed_variable()),
                 extend=lambda s: one_of(build_binoperator_strat(s)(),
                                         build_unoperator_strat(s)(),
                                         build_reduce_strat(s)()),
                 max_leaves=8))
def test_convert_ineq_identity(expr: NumericExpression):
    expr_convert = convert_expr_to_ineq((expr == expr).get_data())
    assert(isinstance(expr_convert, IneqData))


@given(recursive(base=one_of(build_indexed_parameter(),
                             build_indexed_variable()),
                 extend=lambda s: one_of(build_binoperator_strat(s)(),
                                         build_unoperator_strat(s)(),
                                         build_reduce_strat(s)()),
                 max_leaves=8))
def test_convert_expr_to_ineq_fail(expr: NumericExpression):
    assume(not isinstance(expr, Reduce))
    with pytest.raises(TypeError):
        convert_expr_to_ineq(expr.get_data())


@given(recursive(base=one_of(build_indexed_parameter_detailed(),
                             build_indexed_variable_detailed()),
                 extend=lambda s: one_of(build_binoperator_strat_detailed(s)(),
                                         build_unoperator_strat_detailed(s)(),
                                         build_reduce_strat_detailed(s)()),
                 max_leaves=8))
def test_free_indexes(expr: Tuple[NumericExpression,
                                  List[IndexedContainer],
                                  List[IndexedContainer],
                                  FrozenSet[IndexSet]]):
    expr, _, _, index_sets = expr
    free_idx_sets, _ = index_sets
    assert(free_index_sets(expr.get_data()) == free_idx_sets)


@given(recursive(base=one_of(build_indexed_parameter_detailed(),
                             build_indexed_variable_detailed()),
                 extend=lambda s: one_of(build_binoperator_strat_detailed(s)(),
                                         build_unoperator_strat_detailed(s)(),
                                         build_reduce_strat_detailed(s)()),
                 max_leaves=8))
def test_reduce_indexes(expr: Tuple[NumericExpression,
                                    List[IndexedContainer],
                                    List[IndexedContainer],
                                    FrozenSet[IndexSet]]):
    expr, _, _, index_sets = expr
    _, reduce_idx_sets = index_sets
    assert(reduce_index_sets(expr.get_data()) == reduce_idx_sets)


@given(recursive(base=one_of(build_indexed_parameter_detailed(),
                             build_indexed_variable_detailed()),
                 extend=lambda s: one_of(build_binoperator_strat_detailed(s)(),
                                         build_unoperator_strat_detailed(s)(),
                                         build_reduce_strat_detailed(s)()),
                 max_leaves=8),
       recursive(base=one_of(build_indexed_parameter_detailed(),
                             build_indexed_variable_detailed()),
                 extend=lambda s: one_of(build_binoperator_strat_detailed(s)(),
                                         build_unoperator_strat_detailed(s)(),
                                         build_reduce_strat_detailed(s)()),
                 max_leaves=8),
       sampled_from(Ineqoperator))
def test_free_indexes_ineq(expr: Tuple[NumericExpression,
                                       List[IndexedContainer],
                                       List[IndexedContainer],
                                       FrozenSet[IndexSet]],
                           expr_2: Tuple[NumericExpression,
                                         List[IndexedContainer],
                                         List[IndexedContainer],
                                         FrozenSet[IndexSet]],
                           ineq_op: Ineqoperator):
    expr, _, _, index_sets = expr
    expr_2, _, _, index_sets_2 = expr_2
    free_idx_sets, _ = index_sets
    free_idx_sets_2, _ = index_sets_2
    ineq_expr = ineq_op.value(expr, expr_2)
    assert(free_index_sets(ineq_expr.get_data()) ==
           free_idx_sets.union(free_idx_sets_2))


@given(recursive(base=one_of(build_indexed_parameter_detailed(),
                             build_indexed_variable_detailed()),
                 extend=lambda s: one_of(build_binoperator_strat_detailed(s)(),
                                         build_unoperator_strat_detailed(s)(),
                                         build_reduce_strat_detailed(s)()),
                 max_leaves=8),
       recursive(base=one_of(build_indexed_parameter_detailed(),
                             build_indexed_variable_detailed()),
                 extend=lambda s: one_of(build_binoperator_strat_detailed(s)(),
                                         build_unoperator_strat_detailed(s)(),
                                         build_reduce_strat_detailed(s)()),
                 max_leaves=8),
       sampled_from(Ineqoperator))
def test_reduce_indexes_ineq(expr: Tuple[NumericExpression,
                                         List[IndexedContainer],
                                         List[IndexedContainer],
                                         FrozenSet[IndexSet]],
                             expr_2: Tuple[NumericExpression,
                                           List[IndexedContainer],
                                           List[IndexedContainer],
                                           FrozenSet[IndexSet]],
                             ineq_op: Ineqoperator):
    expr, _, _, index_sets = expr
    expr_2, _, _, index_sets_2 = expr_2
    _, reduce_idx_sets = index_sets
    _, reduce_idx_sets_2 = index_sets_2
    ineq_expr = ineq_op.value(expr, expr_2)
    assert(reduce_index_sets(ineq_expr.get_data()) ==
           reduce_idx_sets.union(reduce_idx_sets_2))


def test_free_indexes_expr_fail():
    with pytest.raises(NotImplementedError):
        free_index_sets(ExpressionData())


def test_reduce_indexes_expr_fail():
    with pytest.raises(NotImplementedError):
        reduce_index_sets(ExpressionData())


@given(recursive(base=one_of(build_indexed_parameter_detailed(),
                             build_indexed_variable_detailed()),
                 extend=lambda s: one_of(build_binoperator_strat_detailed(s)(),
                                         build_unoperator_strat_detailed(s)(),
                                         build_reduce_strat_detailed(s)()),
                 max_leaves=8))
def test_get_variables(expr: Tuple[NumericExpression,
                                   List[IndexedContainer],
                                   List[IndexedContainer],
                                   FrozenSet[IndexSet]]):
    expr, variables, _, _ = expr
    variables_id = {v.container.id for v in variables}
    assert(get_variables(expr.get_data()) == variables_id)


@given(recursive(base=one_of(build_indexed_parameter_detailed(),
                             build_indexed_variable_detailed()),
                 extend=lambda s: one_of(build_binoperator_strat_detailed(s)(),
                                         build_unoperator_strat_detailed(s)(),
                                         build_reduce_strat_detailed(s)()),
                 max_leaves=8),
       recursive(base=one_of(build_indexed_parameter_detailed(),
                             build_indexed_variable_detailed()),
                 extend=lambda s: one_of(build_binoperator_strat_detailed(s)(),
                                         build_unoperator_strat_detailed(s)(),
                                         build_reduce_strat_detailed(s)()),
                 max_leaves=8),
       sampled_from(Ineqoperator))
def test_get_variables_ineq(expr_1: Tuple[NumericExpression,
                                          List[IndexedContainer],
                                          List[IndexedContainer],
                                          FrozenSet[IndexSet]],
                            expr_2: Tuple[NumericExpression,
                                          List[IndexedContainer],
                                          List[IndexedContainer],
                                          FrozenSet[IndexSet]],
                            ineq_op: Ineqoperator):
    expr_1, variables_1, _, _ = expr_1
    expr_2, variables_2, _, _ = expr_2
    expr = ineq_op.value(expr_1, expr_2)
    variables_1_id = {v.container.id for v in variables_1}
    variables_2_id = {v.container.id for v in variables_2}
    variables_id = variables_1_id.union(variables_2_id)
    assert(get_variables(expr.get_data()) == variables_id)


def test_get_variables_expr_fail():
    with pytest.raises(NotImplementedError):
        get_variables(ExpressionData())


@given(recursive(base=one_of(build_indexed_parameter_detailed(),
                             build_indexed_variable_detailed()),
                 extend=lambda s: one_of(build_binoperator_strat_detailed(s)(),
                                         build_unoperator_strat_detailed(s)(),
                                         build_reduce_strat_detailed(s)()),
                 max_leaves=8))
def test_get_parameters(expr: Tuple[NumericExpression,
                                    List[IndexedContainer],
                                    List[IndexedContainer],
                                    FrozenSet[IndexSet]]):
    expr, _, parameters, _ = expr
    parameters_id = {p.container.id for p in parameters}
    assert(get_parameters(expr.get_data()) == parameters_id)


@given(recursive(base=one_of(build_indexed_parameter_detailed(),
                             build_indexed_variable_detailed()),
                 extend=lambda s: one_of(build_binoperator_strat_detailed(s)(),
                                         build_unoperator_strat_detailed(s)(),
                                         build_reduce_strat_detailed(s)()),
                 max_leaves=8),
       recursive(base=one_of(build_indexed_parameter_detailed(),
                             build_indexed_variable_detailed()),
                 extend=lambda s: one_of(build_binoperator_strat_detailed(s)(),
                                         build_unoperator_strat_detailed(s)(),
                                         build_reduce_strat_detailed(s)()),
                 max_leaves=8),
       sampled_from(Ineqoperator))
def test_get_parameters_ineq(expr_1: Tuple[NumericExpression,
                                           List[IndexedContainer],
                                           List[IndexedContainer],
                                           FrozenSet[IndexSet]],
                             expr_2: Tuple[NumericExpression,
                                           List[IndexedContainer],
                                           List[IndexedContainer],
                                           FrozenSet[IndexSet]],
                             ineq_op: Ineqoperator):
    expr_1, _, parameters_1, _ = expr_1
    expr_2, _, parameters_2, _ = expr_2
    expr = ineq_op.value(expr_1, expr_2)
    parameters_1_id = {p.container.id for p in parameters_1}
    parameters_2_id = {p.container.id for p in parameters_2}
    parameters_id = parameters_1_id.union(parameters_2_id)
    assert(get_parameters(expr.get_data()) == parameters_id)


def test_get_parameters_expr_fail():
    with pytest.raises(NotImplementedError):
        get_parameters(ExpressionData())


@given(recursive(base=one_of(build_indexed_parameter(),
                             build_indexed_variable()),
                 extend=lambda s: one_of(build_binoperator_strat(s)(),
                                         build_unoperator_strat(s)(),
                                         build_reduce_strat(s)()),
                 max_leaves=8))
def test_get_expr_from_numerical_expr_data(expr: NumericExpression):
    expr_data = expr.get_data()
    expr_from_data = get_expr_from_data(expr_data)
    assert(expr_data == expr_from_data.get_data())


@given(recursive(base=one_of(build_indexed_parameter(),
                             build_indexed_variable()),
                 extend=lambda s: one_of(build_binoperator_strat(s)(),
                                         build_unoperator_strat(s)(),
                                         build_reduce_strat(s)()),
                 max_leaves=8),
       recursive(base=one_of(build_indexed_parameter(),
                             build_indexed_variable()),
                 extend=lambda s: one_of(build_binoperator_strat(s)(),
                                         build_unoperator_strat(s)(),
                                         build_reduce_strat(s)()),
                 max_leaves=8),
       sampled_from(Ineqoperator))
def test_get_expr_from_ineq_expr_data(expr: NumericExpression,
                                      expr_2: NumericExpression,
                                      ineq_op: Ineqoperator):
    expr_data = expr.get_data()
    expr_from_data = get_expr_from_data(expr_data)
    assert(expr_data == expr_from_data.get_data())


def test_get_expr_from_data_fail():
    with pytest.raises(NotImplementedError):
        get_expr_from_data(ExpressionData())
