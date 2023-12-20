from uliege.decision_process.decision_process_components.variable import Type
from uliege.decision_process.decision_process_components.expression.reduce import Reduce,\
                                                          Reducer,\
                                                          ReduceIndexSet
from uliege.decision_process.decision_process_components.expression.numeric_expression import\
    NumericExpression
from uliege.decision_process.decision_process_components.evaluator import Evaluator
from hypothesis import assume, given
from hypothesis.strategies import sampled_from,\
                                  one_of,\
                                  recursive,\
                                  floats
import random
import numpy as np
from typing import Dict, Tuple
from .utils import build_binoperator_eval_strat,\
                        build_idx_container_evaluator,\
                        build_unoperator_eval_strat
from uliege.decision_process.decision_process_components.cost_function import CostFunction,\
                                                      LastTimeStepMask,\
                                                      HorizonMask
from uliege.decision_process.utils.utils import free_index_sets, get_variables
from math import isnan

np.random.seed(1000)
random.seed(1000)


@given(recursive(base=build_idx_container_evaluator(),
                 extend=lambda s: one_of(build_binoperator_eval_strat(s)(),
                                         build_unoperator_eval_strat(s)()),
                 max_leaves=2),
       sampled_from(Reducer))
def test_execution_costfunction(expr_eval: Tuple[NumericExpression,
                                                 Evaluator,
                                                 Dict[Type, float],
                                                 float],
                                reducer: Reducer):
    expr, eval, _, _ = expr_eval
    assume(len(get_variables(expr.get_data())) > 0)
    idx_sets = free_index_sets(expr.get_data())
    for idx_set in idx_sets:
        expr = Reduce(expr, ReduceIndexSet(reducer=reducer,
                                           idx_set=idx_set))
    c = CostFunction(cost_expression=expr)
    assert(c(eval) == expr(eval))


@given(recursive(base=build_idx_container_evaluator(),
                 extend=lambda s: one_of(build_binoperator_eval_strat(s)(),
                                         build_unoperator_eval_strat(s)()),
                 max_leaves=2),
       sampled_from(Reducer),
       floats(min_value=0,
              allow_nan=None,
              allow_infinity=None,
              exclude_min=True,
              max_value=100),
       floats(min_value=0,
              allow_nan=None,
              allow_infinity=None,
              exclude_min=True,
              max_value=100))
def test_execution_costfunction_reverse_mask_params(
    expr_eval: Tuple[NumericExpression,
                     Evaluator,
                     Dict[Type, float],
                     float], reducer: Reducer, t: float, T: float):
    expr, eval, _, _ = expr_eval
    assume(len(get_variables(expr.get_data())) > 0)
    idx_sets = free_index_sets(expr.get_data())
    for idx_set in idx_sets:
        expr = Reduce(expr, ReduceIndexSet(reducer=reducer,
                                           idx_set=idx_set))

    class CustomMask(HorizonMask):

        def __call__(self, t: float, T: float):
            return t/T
    c = CostFunction(cost_expression=expr,
                     horizon_mask=CustomMask())
    assume(not isnan(c(eval, t=t, T=T)))
    assume(not isnan(c(eval, t=T, T=t)))
    assert(c(eval, t=t, T=T) == c(eval, t=T, T=t))


@given(recursive(base=build_idx_container_evaluator(),
                 extend=lambda s: one_of(build_binoperator_eval_strat(s)(),
                                         build_unoperator_eval_strat(s)()),
                 max_leaves=2),
       sampled_from(Reducer))
def test_execution_costfunction_sparse_mask(expr_eval: Tuple[NumericExpression,
                                                             Evaluator,
                                                             Dict[Type, float],
                                                             float],
                                            reducer: Reducer):
    expr, eval, _, _ = expr_eval
    assume(len(get_variables(expr.get_data())) > 0)
    idx_sets = free_index_sets(expr.get_data())
    for idx_set in idx_sets:
        expr = Reduce(expr, ReduceIndexSet(reducer=reducer,
                                           idx_set=idx_set))
    c = CostFunction(cost_expression=expr,
                     horizon_mask=LastTimeStepMask())
    assume(not isnan(c(eval)))
    assert(c(eval) == 0)
