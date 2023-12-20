import pytest
from uliege.decision_process.decision_process_components.expression.indexed_container import\
                                                              IndexedContainer
from uliege.decision_process.decision_process_components.variable import Variable,\
                                                 Type
from uliege.decision_process.decision_process_components.parameter import Parameter
from uliege.decision_process.decision_process_components.expression.index_set import\
                                                 IndexSet
from uliege.decision_process.decision_process_components.expression.binop import Binoperator
from uliege.decision_process.decision_process_components.expression.unop import Unoperator
from uliege.decision_process.decision_process_components.expression.reduce import Reduce,\
                                                          Reducer,\
                                                          ReduceIndexSet,\
                                                          sum_reduce,\
                                                          prod_reduce,\
                                                          OverriderEvaluator
from uliege.decision_process.decision_process_components.expression.numeric_expression import\
    NumericExpression
from uliege.decision_process.decision_process_components.expression.ineq import Ineqoperator
from uliege.decision_process.decision_process_components.evaluator import Evaluator,\
                                                  ValueNotFoundError,\
                                                  IndexSetNotFoundError
from hypothesis import assume, given
from .utils import reasonable_lenghty_text, SimpleEvaluator
from hypothesis.strategies import sampled_from,\
                                  one_of,\
                                  recursive
import random
import numpy as np
from typing import Dict, Tuple
from .utils import build_binoperator_eval_strat,\
                        build_idx_container_evaluator,\
                        build_reduce_eval_strat,\
                        build_unoperator_eval_strat,\
                        combine_evaluators

np.random.seed(1000)
random.seed(1000)


@given(build_idx_container_evaluator())
def test_evaluation_ic(ic_eval: Tuple[IndexedContainer,
                                      Evaluator,
                                      Dict[Type, float],
                                      float]):
    idx_container, evaluator, map_type_to_value, val_param = ic_eval
    value_test = val_param
    if isinstance(idx_container.container, Variable):
        v_type = idx_container.container.get_data().v_type
        value_test = map_type_to_value[v_type]
    value_eval = idx_container(evaluator)
    assert(value_test == value_eval)


@given(reasonable_lenghty_text,
       build_idx_container_evaluator())
def test_evaluation_idxst_notfound_fail(suffix: str,
                                        ic_eval: Tuple[IndexedContainer,
                                                       Evaluator,
                                                       Dict[Type, float],
                                                       float]):
    idx_container, evaluator, _, _ = ic_eval
    idx_seq_uneval = [IndexSet(id=idx.id+suffix,
                               parent=IndexSet(id=idx.id))
                      for idx in idx_container.get_data().idx_seq]
    idx_container_uneval = IndexedContainer(container=idx_container.container,
                                            idx_seq=idx_seq_uneval)
    assume(idx_container_uneval.get_data() != idx_container.get_data())
    with pytest.raises(IndexSetNotFoundError):
        _ = idx_container_uneval(evaluator)


@given(reasonable_lenghty_text,
       build_idx_container_evaluator())
def test_evaluation_cont_notfound_fail(suffix: str,
                                       ic_eval: Tuple[IndexedContainer,
                                                      Evaluator,
                                                      Dict[Type, float],
                                                      float]):
    idx_container, evaluator, _, _ = ic_eval
    idx_seq_eval = idx_container.get_data().container.shape
    new_id = idx_container.container.id
    shape = idx_container.container.shape
    if isinstance(idx_container.container, Variable):
        v_type = idx_container.container.get_data().v_type
        support = idx_container.container.get_data().support
        container_uneval = Variable(id=new_id,
                                    v_type=v_type,
                                    shape=shape,
                                    support=support)
    else:
        container_uneval = Parameter(id=new_id,
                                     shape=shape)
    idx_container_uneval = IndexedContainer(container=container_uneval,
                                            idx_seq=idx_seq_eval)
    with pytest.raises(ValueNotFoundError):
        _ = idx_container_uneval(evaluator)


@given(recursive(base=build_idx_container_evaluator(),
                 extend=lambda s: one_of(build_binoperator_eval_strat(s)(),
                                         build_unoperator_eval_strat(s)(),
                                         build_reduce_eval_strat(s)()),
                 max_leaves=8),
       sampled_from(Unoperator))
def test_evaluation_complex_unoperation(e_eval: Tuple[NumericExpression,
                                                      Evaluator,
                                                      Dict[Type, float],
                                                      float],
                                        unoperator: Unoperator):

    expr, eval, _, _ = e_eval
    eval_expr = unoperator.value(expr)(eval)
    eval_test = unoperator.value(expr(eval))
    assert(eval_test == eval_expr)


@given(recursive(base=build_idx_container_evaluator(),
                 extend=lambda s: one_of(build_binoperator_eval_strat(s)(),
                                         build_unoperator_eval_strat(s)(),
                                         build_reduce_eval_strat(s)()),
                 max_leaves=8),
       recursive(base=build_idx_container_evaluator(),
                 extend=lambda s: one_of(build_binoperator_eval_strat(s)(),
                                         build_unoperator_eval_strat(s)(),
                                         build_reduce_eval_strat(s)()),
                 max_leaves=8),
       sampled_from(Binoperator))
def test_evaluation_complex_binoperation(e_eval_1: Tuple[NumericExpression,
                                                         Evaluator,
                                                         Dict[Type, float],
                                                         float],
                                         e_eval_2: Tuple[NumericExpression,
                                                         Evaluator,
                                                         Dict[Type, float],
                                                         float],
                                         binoperator: Binoperator):

    expr_1, eval_1, map_type, value_parameter = e_eval_1
    expr_2, eval_2, _, _ = e_eval_2
    assume(binoperator != Binoperator.DIV or expr_2(eval_2) != 0)
    eval = combine_evaluators(eval_1, eval_2, map_type, value_parameter)
    eval_expr = binoperator.value(expr_1, expr_2)(eval)
    eval_test = binoperator.value(expr_1(eval_1), expr_2(eval_2))
    assert(eval_test == eval_expr)


@given(recursive(base=build_idx_container_evaluator(),
                 extend=lambda s: one_of(build_binoperator_eval_strat(s)(),
                                         build_unoperator_eval_strat(s)(),
                                         build_reduce_eval_strat(s)()),
                 max_leaves=8))
def test_evaluation_complex_sum(inner_expr_eval: Tuple[NumericExpression,
                                                       SimpleEvaluator,
                                                       Dict[Type, float],
                                                       float]):

    inner_expr, eval, _, _ = inner_expr_eval
    index_set = np.random.choice(list(eval.index_set_map.keys()))
    eval_expr = sum_reduce(inner_expr,
                           idx_set=index_set)(eval)
    eval_test = 0
    for idx in eval.get_all_components_by_index(index_set):
        overrider_evaluator = OverriderEvaluator(eval, (index_set, idx))
        eval_test += inner_expr(overrider_evaluator)
    assert(eval_test == eval_expr)


@given(recursive(base=build_idx_container_evaluator(),
                 extend=lambda s: one_of(build_binoperator_eval_strat(s)(),
                                         build_unoperator_eval_strat(s)(),
                                         build_reduce_eval_strat(s)()),
                 max_leaves=8))
def test_evaluation_complex_prod(inner_expr_eval: Tuple[NumericExpression,
                                                        SimpleEvaluator,
                                                        Dict[Type, float],
                                                        float]):

    inner_expr, eval, _, _ = inner_expr_eval
    index_set = np.random.choice(list(eval.index_set_map.keys()))
    eval_expr = prod_reduce(inner_expr,
                            idx_set=index_set)(eval)
    eval_test = 1
    for idx in eval.get_all_components_by_index(index_set):
        overrider_evaluator = OverriderEvaluator(eval, (index_set, idx))
        eval_test *= inner_expr(overrider_evaluator)
    assert(eval_test == eval_expr)


@given(recursive(base=build_idx_container_evaluator(),
                 extend=lambda s: one_of(build_binoperator_eval_strat(s)(),
                                         build_unoperator_eval_strat(s)(),
                                         build_reduce_eval_strat(s)()),
                 max_leaves=8),
       reasonable_lenghty_text,
       sampled_from(Reducer))
def test_evaluation_reduce_idxset_notfound(inner_expr: Tuple[NumericExpression,
                                                             SimpleEvaluator,
                                                             Dict[Type, float],
                                                             float],
                                           idx_set_name: str,
                                           reducer: Reducer):

    inner_expr, eval, _, _ = inner_expr
    idx_set = IndexSet(id=idx_set_name)
    assume(idx_set not in eval.index_set_map.keys())
    eval_expr = Reduce(inner_expr, ReduceIndexSet(reducer=reducer,
                                                  idx_set=idx_set))
    with pytest.raises(IndexSetNotFoundError):
        eval_expr(eval)


@given(recursive(base=build_idx_container_evaluator(),
                 extend=lambda s: one_of(build_binoperator_eval_strat(s)(),
                                         build_unoperator_eval_strat(s)(),
                                         build_reduce_eval_strat(s)()),
                 max_leaves=8),
       recursive(base=build_idx_container_evaluator(),
                 extend=lambda s: one_of(build_binoperator_eval_strat(s)(),
                                         build_unoperator_eval_strat(s)(),
                                         build_reduce_eval_strat(s)()),
                 max_leaves=8),
       sampled_from(Ineqoperator))
def test_evaluation_complex_inequation(e_eval_1: Tuple[NumericExpression,
                                                       Evaluator,
                                                       Dict[Type, float],
                                                       float],
                                       e_eval_2: Tuple[NumericExpression,
                                                       Evaluator,
                                                       Dict[Type, float],
                                                       float],
                                       ineqoperator: Ineqoperator):

    expr_1, eval_1, map_type, value_parameter = e_eval_1
    expr_2, eval_2, _, _ = e_eval_2
    eval = combine_evaluators(eval_1, eval_2, map_type, value_parameter)
    eval_expr = ineqoperator.value(expr_1, expr_2)(eval)
    eval_test = ineqoperator.value(expr_1(eval_1), expr_2(eval_2))
    assert(eval_test == eval_expr)
