import pytest
from uliege.decision_process.decision_process_components.variable import NoVariableInvolvedError
from uliege.decision_process.decision_process_components.expression.reduce import Reduce,\
                                                          Reducer,\
                                                          ReduceIndexSet
from hypothesis import assume, given
from hypothesis.strategies import sampled_from,\
                                  one_of,\
                                  recursive
import random
import numpy as np
from .utils import build_indexed_parameter,\
                        build_indexed_variable,\
                        build_binoperator_strat,\
                        build_unoperator_strat,\
                        build_reduce_strat
from uliege.decision_process.decision_process_components.expression.numeric_expression import\
    NumericExpression
from uliege.decision_process.decision_process_components.cost_function import CostFunction,\
                                                      FreeIndexError,\
                                                      UniformStepMask
from uliege.decision_process.utils.utils import get_variables, free_index_sets

np.random.seed(1000)
random.seed(1000)


@given(recursive(base=one_of(build_indexed_parameter(),
                             build_indexed_variable()),
                 extend=lambda s: one_of(build_binoperator_strat(s)(),
                                         build_unoperator_strat(s)(),
                                         build_reduce_strat(s)()),
                 max_leaves=8),
       sampled_from(Reducer))
def test_valid_costfunction(expr: NumericExpression,
                            reducer: Reducer):
    assume(len(get_variables(expr.get_data())) > 0)
    idx_sets = free_index_sets(expr.get_data())
    for idx_set in idx_sets:
        expr = Reduce(expr, ReduceIndexSet(reducer=reducer,
                                           idx_set=idx_set))
    c = CostFunction(cost_expression=expr)
    assert(c.cost_expression.get_data() == expr.get_data())
    assert(c.horizon_mask(np.random.rand(), np.random.rand()))


@given(recursive(base=build_indexed_parameter(),
                 extend=lambda s: one_of(build_binoperator_strat(s)(),
                                         build_unoperator_strat(s)(),
                                         build_reduce_strat(s)()),
                 max_leaves=8),
       sampled_from(Reducer))
def test_costfunction_without_variables_fail(expr: NumericExpression,
                                             reducer: Reducer):
    idx_sets = free_index_sets(expr.get_data())
    for idx_set in idx_sets:
        expr = Reduce(expr, ReduceIndexSet(reducer=reducer,
                                           idx_set=idx_set))
    with pytest.raises(NoVariableInvolvedError):
        _ = CostFunction(cost_expression=expr)


@given(recursive(base=one_of(build_indexed_parameter(),
                             build_indexed_variable()),
                 extend=lambda s: one_of(build_binoperator_strat(s)(),
                                         build_unoperator_strat(s)(),
                                         build_reduce_strat(s)()),
                 max_leaves=8),
       sampled_from(Reducer))
def test_costfunction_with_free_indexes_fail(expr: NumericExpression,
                                             reducer: Reducer):
    assume(len(get_variables(expr.get_data())) > 0)
    idx_sets = free_index_sets(expr.get_data())
    assume(len(idx_sets) > 0)
    with pytest.raises(FreeIndexError):
        _ = CostFunction(cost_expression=expr)


@given(recursive(base=one_of(build_indexed_parameter(),
                             build_indexed_variable()),
                 extend=lambda s: one_of(build_binoperator_strat(s)(),
                                         build_unoperator_strat(s)(),
                                         build_reduce_strat(s)()),
                 max_leaves=8),
       sampled_from(Reducer))
def test_costfunction_from_data(expr: NumericExpression,
                                reducer: Reducer):
    assume(len(get_variables(expr.get_data())) > 0)
    idx_sets = free_index_sets(expr.get_data())
    for idx_set in idx_sets:
        expr = Reduce(expr, ReduceIndexSet(reducer=reducer,
                                           idx_set=idx_set))
    c = CostFunction(cost_expression=expr)

    cclone = CostFunction.from_data(c.get_data())
    cdata = c.get_data()
    cclonedata = cclone.get_data()
    assert(cclonedata.cost_expression ==
           cdata.cost_expression
           and cclonedata.id == cdata.id
           and isinstance(cdata.horizon_mask, UniformStepMask)
           and isinstance(cclonedata.horizon_mask, UniformStepMask))
