from uliege.decision_process.decision_process_components.variable import Type
from uliege.decision_process.decision_process_components.expression.numeric_expression import\
    NumericExpression
from uliege.decision_process.decision_process_components.expression.ineq import Ineqoperator
from uliege.decision_process.decision_process_components.evaluator import Evaluator
from hypothesis import assume, given
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
from uliege.decision_process.decision_process_components.constraint import Constraint
from uliege.decision_process.utils.utils import get_variables

np.random.seed(1000)
random.seed(1000)


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
def test_execution_constraint(e_eval_1: Tuple[NumericExpression,
                                              Evaluator,
                                              Dict[Type, float],
                                              float],
                              e_eval_2: Tuple[NumericExpression,
                                              Evaluator,
                                              Dict[Type, float],
                                              float],
                              ineq_op: Ineqoperator):
    expr_1, eval_1, map_type, value_param = e_eval_1
    expr_2, eval_2, _, _ = e_eval_2
    evalu = combine_evaluators(eval_1, eval_2, map_type, value_param)
    expr = ineq_op.value(expr_1, expr_2)
    assume(len(get_variables(expr.get_data())) > 0)
    c = Constraint(ineq=expr)
    assert(c(evalu) == expr(evalu))
