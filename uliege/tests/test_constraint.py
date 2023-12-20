import sys

import pytest
from uliege.decision_process import NoVariableInvolvedError
from uliege.decision_process import Ineqoperator
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
from uliege.decision_process import NumericExpression
from uliege.decision_process import Constraint
from uliege.decision_process import get_variables

np.random.seed(1000)
random.seed(1000)


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
def test_valid_constraint(expr_1: NumericExpression,
                          expr_2: NumericExpression,
                          ineq_op: Ineqoperator):
    ineq_expr = ineq_op.value(expr_1, expr_2)
    assume(len(get_variables(ineq_expr.get_data())) > 0)
    c = Constraint(ineq=ineq_expr)
    assert(c.ineq.get_data() == ineq_expr.get_data())


@given(recursive(base=build_indexed_parameter(),
                 extend=lambda s: one_of(build_binoperator_strat(s)(),
                                         build_unoperator_strat(s)(),
                                         build_reduce_strat(s)()),
                 max_leaves=8),
       recursive(base=build_indexed_parameter(),
                 extend=lambda s: one_of(build_binoperator_strat(s)(),
                                         build_unoperator_strat(s)(),
                                         build_reduce_strat(s)()),
                 max_leaves=8),
       sampled_from(Ineqoperator))
def test_constraint_no_variable(expr_1: NumericExpression,
                                expr_2: NumericExpression,
                                ineq_op: Ineqoperator):
    ineq_expr = ineq_op.value(expr_1, expr_2)
    with pytest.raises(NoVariableInvolvedError):
        _ = Constraint(ineq=ineq_expr)


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
def test_constraint_from_data(expr_1: NumericExpression,
                              expr_2: NumericExpression,
                              ineq_op: Ineqoperator):
    ineq_expr = ineq_op.value(expr_1, expr_2)
    assume(len(get_variables(ineq_expr.get_data())) > 0)
    c = Constraint(ineq=ineq_expr)
    cdata = c.get_data()
    cfromdata = Constraint.from_data(cdata)
    assert(cdata == cfromdata.get_data())
