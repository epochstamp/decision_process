import pytest
from uliege.decision_process.decision_process_components.expression.indexed_container import\
    IndexedContainer, IndexedVariableData
from hypothesis import assume, given
from hypothesis.strategies import one_of,\
                                  recursive
import random
import numpy as np
from .utils import build_indexed_parameter,\
                        build_indexed_variable,\
                        build_binoperator_strat,\
                        build_unoperator_strat,\
                        build_reduce_strat,\
                        build_scalar_variable
from uliege.decision_process.decision_process_components.expression.numeric_expression import\
    NumericExpression
from uliege.decision_process.decision_process_components.dynamics import Dynamics

np.random.seed(1000)
random.seed(1000)


@given(build_indexed_variable(),
       recursive(base=one_of(build_indexed_parameter(),
                             build_indexed_variable()),
                 extend=lambda s: one_of(build_binoperator_strat(s)(),
                                         build_unoperator_strat(s)(),
                                         build_reduce_strat(s)()),
                 max_leaves=8)
       )
def test_valid_dynamics_on_indexed_var(var: IndexedContainer,
                                       var_update: NumericExpression):
    d = Dynamics(state_var=var, state_var_update=var_update)
    assert(d.get_data().state_var == var.get_data()
           and d.get_data().state_var_update == var_update.get_data())


@given(build_scalar_variable(),
       recursive(base=one_of(build_indexed_parameter(),
                             build_indexed_variable()),
                 extend=lambda s: one_of(build_binoperator_strat(s)(),
                                         build_unoperator_strat(s)(),
                                         build_reduce_strat(s)()),
                 max_leaves=8)
       )
def test_valid_dynamics_on_scalar_var(var: IndexedContainer,
                                      var_update: NumericExpression):
    d = Dynamics(state_var=var, state_var_update=var_update)
    assert(isinstance(d.get_data().state_var, IndexedVariableData))
    assert(d.get_data().state_var.container == var.container.get_data()
           and d.get_data().state_var_update == var_update.get_data())


@given(build_indexed_variable(),
       recursive(base=one_of(build_indexed_parameter(),
                             build_indexed_variable()),
                 extend=lambda s: one_of(build_binoperator_strat(s)(),
                                         build_unoperator_strat(s)(),
                                         build_reduce_strat(s)()),
                 max_leaves=8)
       )
def test_dynamics_not_fully_indexed_var_fail(var: IndexedContainer,
                                             var_update: NumericExpression):
    container = var.container
    assume(len(container.shape) > 0)

    with pytest.raises(ValueError):
        _ = Dynamics(state_var=container,
                     state_var_update=var_update)


@given(build_indexed_parameter(),
       recursive(base=one_of(build_indexed_parameter(),
                             build_indexed_variable()),
                 extend=lambda s: one_of(build_binoperator_strat(s)(),
                                         build_unoperator_strat(s)(),
                                         build_reduce_strat(s)()),
                 max_leaves=8)
       )
def test_dynamics_indexed_parameter_fail(param: IndexedContainer,
                                         var_update: NumericExpression):
    with pytest.raises(TypeError):
        _ = Dynamics(state_var=param,
                     state_var_update=var_update)


@given(build_indexed_variable(),
       recursive(base=one_of(build_indexed_parameter(),
                             build_indexed_variable()),
                 extend=lambda s: one_of(build_binoperator_strat(s)(),
                                         build_unoperator_strat(s)(),
                                         build_reduce_strat(s)()),
                 max_leaves=8)
       )
def test_dynamics_from_data_on_indexed_var(var: IndexedContainer,
                                           var_update: NumericExpression):
    d = Dynamics(state_var=var, state_var_update=var_update)
    d_from_data = Dynamics.from_data(d.get_data())
    assert(d.get_data() == d_from_data.get_data())
