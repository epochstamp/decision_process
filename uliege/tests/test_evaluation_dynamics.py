from uliege.decision_process.decision_process_components.expression.indexed_container import\
                                                              IndexedContainer
from uliege.decision_process.decision_process_components.variable import Type
from uliege.decision_process.decision_process_components.expression.numeric_expression import\
    NumericExpression
from uliege.decision_process.decision_process_components.evaluator import Evaluator
from hypothesis import given
from hypothesis.strategies import one_of,\
                                  recursive
import random
import numpy as np
from typing import Dict, Tuple
from .utils import build_binoperator_eval_strat,\
                        build_idx_container_evaluator,\
                        build_indexed_variable,\
                        build_reduce_eval_strat,\
                        build_unoperator_eval_strat
from uliege.decision_process.decision_process_components.dynamics import Dynamics

np.random.seed(1000)
random.seed(1000)


@given(build_indexed_variable(),
       recursive(base=build_idx_container_evaluator(),
                 extend=lambda s: one_of(build_binoperator_eval_strat(s)(),
                                         build_unoperator_eval_strat(s)(),
                                         build_reduce_eval_strat(s)()),
                 max_leaves=8))
def test_execution_dynamics(ic: IndexedContainer,
                            e_eval: Tuple[NumericExpression,
                                          Evaluator,
                                          Dict[Type, float],
                                          float]):
    expr, eval, _, _ = e_eval
    d = Dynamics(ic, expr)
    assert(d(eval) == expr(eval))
