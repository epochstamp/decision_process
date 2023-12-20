import traceback
import pytest
from uliege.decision_process import IndexedContainer
from uliege.decision_process import (
    Variable,
    Parameter,
    Binoperator,
    Ineqoperator,
    Dynamics,
    Constraint,
    CostFunction,
    DecisionProcess
)
from hypothesis import HealthCheck, given, settings, assume
from hypothesis.strategies import sampled_from
import random
from .utils import build_decision_process
from uliege.decision_process.decision_process import (
    UselessVariableError,
    UselessParameterError,
    DynamicsLessStateVariableError,
    NoCostFunctionError,
    AlreadyExistsError,
    NotAStateVariableError,
    NotDefinedVariableError,
    NotDefinedParameterError
)


@given(build_decision_process())
@settings(suppress_health_check=(HealthCheck.too_slow,
                                 HealthCheck.large_base_example))
def test_valid_decision_process(decision_process: DecisionProcess):
    try:
        decision_process.validate()
        assert(decision_process.validated)
    except BaseException:
        traceback.print_exc()
        assert(False)


@given(build_decision_process())
@settings(suppress_health_check=(HealthCheck.too_slow,
                                 HealthCheck.large_base_example))
def test_valid_decision_process_fully_known(decision_process: DecisionProcess):
    assert(decision_process.is_fully_known())


def test_valid_empty_decision_process():
    try:
        assert(DecisionProcess().validate())
    except BaseException:
        traceback.print_exc()
        assert(False)


@given(build_decision_process())
@settings(suppress_health_check=(HealthCheck.too_slow,
                                 HealthCheck.large_base_example))
def test_decision_process_useless_var_fail(decision_process: DecisionProcess):
    decision_process.add_helper_variables(Variable())
    with pytest.raises(UselessVariableError):
        decision_process.validate()


@settings(suppress_health_check=(HealthCheck.too_slow,
                                 HealthCheck.large_base_example))
@given(build_decision_process())
def test_decision_process_unused_param_fail(decision_process: DecisionProcess):
    decision_process.add_parameters(Parameter())
    with pytest.raises(UselessParameterError):
        decision_process.validate()


@settings(suppress_health_check=(HealthCheck.too_slow,
                                 HealthCheck.large_base_example))
@given(build_decision_process())
def test_decision_process_static_state_fail(decision_process: DecisionProcess):
    v = Variable()
    p = Parameter()
    decision_process.add_state_variables(v)
    decision_process.add_parameters(p)
    decision_process.add_constraint_functions(Constraint(v <= p))
    with pytest.raises(DynamicsLessStateVariableError):
        decision_process.validate()


@settings(suppress_health_check=(HealthCheck.too_slow,
                                 HealthCheck.large_base_example))
@given(build_decision_process())
def test_decision_process_no_state_var_fail(decision_process: DecisionProcess):
    not_state_vars = decision_process.action_variables +\
                                      decision_process.helper_variables
    if len(not_state_vars) == 0:
        v = Variable()
        decision_process.add_action_variables(v)
    else:
        v = random.sample(not_state_vars, 1)[0]
    p = Parameter()
    decision_process.add_parameters(p)
    v = IndexedContainer(v)
    with pytest.raises(NotAStateVariableError):
        decision_process.add_dynamics_functions(Dynamics(state_var=v,
                                                         state_var_update=v+p))


@settings(suppress_health_check=(HealthCheck.too_slow,
                                 HealthCheck.large_base_example))
@given(build_decision_process())
def test_decision_process_no_cost_func_fail(decision_process: DecisionProcess):
    decision_process_nocost = DecisionProcess()
    decision_process_nocost.add_state_variables(
        *decision_process.state_variables)
    decision_process_nocost.add_action_variables(
        *decision_process.action_variables)
    decision_process_nocost.add_helper_variables(
        *decision_process.helper_variables)
    decision_process_nocost.add_parameters(
        *decision_process.parameters)
    decision_process_nocost.add_dynamics_functions(
        *decision_process.dynamics_functions)
    decision_process_nocost.add_constraint_functions(
        *decision_process.constraint_functions)
    with pytest.raises(NoCostFunctionError):
        decision_process_nocost.validate()


def test_decision_process_add_twice_variables():
    decision_process = DecisionProcess()
    v = Variable()
    decision_process.add_state_variables(v)
    with pytest.raises(AlreadyExistsError):
        decision_process.add_state_variables(v)
    with pytest.raises(AlreadyExistsError):
        decision_process.add_action_variables(v)
    with pytest.raises(AlreadyExistsError):
        decision_process.add_helper_variables(v)
    decision_process = DecisionProcess()
    decision_process.add_action_variables(v)
    with pytest.raises(AlreadyExistsError):
        decision_process.add_state_variables(v)
    with pytest.raises(AlreadyExistsError):
        decision_process.add_action_variables(v)
    with pytest.raises(AlreadyExistsError):
        decision_process.add_helper_variables(v)
    decision_process = DecisionProcess()
    decision_process.add_helper_variables(v)
    with pytest.raises(AlreadyExistsError):
        decision_process.add_state_variables(v)
    with pytest.raises(AlreadyExistsError):
        decision_process.add_action_variables(v)
    with pytest.raises(AlreadyExistsError):
        decision_process.add_helper_variables(v)


def test_decision_process_add_twice_parameters():
    decision_process = DecisionProcess()
    p = Parameter()
    p2 = Parameter()
    decision_process.add_parameters(p)
    with pytest.raises(AlreadyExistsError):
        decision_process.add_parameters(p, p2)


@given(sampled_from(Binoperator))
def test_decision_process_add_twice_dynamics(binop: Binoperator):
    decision_process = DecisionProcess()
    v = Variable()
    p = Parameter()
    decision_process.add_state_variables(v)
    decision_process.add_parameters(p)
    d = Dynamics(state_var=v, state_var_update=binop.value(v, p))
    decision_process.add_dynamics_functions(d)
    with pytest.raises(AlreadyExistsError):
        decision_process.add_dynamics_functions(d)


@given(sampled_from(Ineqoperator))
def test_decision_process_add_twice_constraint(ineq_op: Ineqoperator):
    decision_process = DecisionProcess()
    v = Variable()
    p = Parameter()
    decision_process.add_state_variables(v)
    decision_process.add_parameters(p)
    c = Constraint(ineq=ineq_op.value(v, p))
    decision_process.add_constraint_functions(c)
    with pytest.raises(AlreadyExistsError):
        decision_process.add_constraint_functions(c)


@given(sampled_from(Binoperator))
def test_decision_process_add_twice_costfunction(binop: Binoperator):
    decision_process = DecisionProcess()
    v = Variable()
    p = Parameter()
    decision_process.add_state_variables(v)
    decision_process.add_parameters(p)
    c = CostFunction(binop.value(v, p))
    decision_process.add_cost_functions(c)
    with pytest.raises(AlreadyExistsError):
        decision_process.add_cost_functions(c)


def test_decision_process_add_useless_var():
    decision_process = DecisionProcess()
    v = Variable()
    v2 = Variable()
    p = Parameter()
    decision_process.add_helper_variables(v)
    decision_process.add_helper_variables(v2)
    decision_process.add_parameters(p)
    c = CostFunction(v+p)
    decision_process.add_cost_functions(c)
    with pytest.raises(UselessVariableError):
        decision_process.validate()


def test_decision_process_add_useless_param():
    decision_process = DecisionProcess()
    v = Variable()
    v2 = Variable()
    p = Parameter()
    decision_process.add_helper_variables(v)
    decision_process.add_helper_variables(v2)
    decision_process.add_parameters(p)
    c = CostFunction(v+v2)
    decision_process.add_cost_functions(c)
    with pytest.raises(UselessParameterError):
        decision_process.validate()


def test_decision_process_undefined_var():
    decision_process = DecisionProcess()
    v = Variable()
    v2 = Variable()
    c = CostFunction(v+v2)
    with pytest.raises(NotDefinedVariableError):
        decision_process.add_cost_functions(c)


def test_decision_process_undefined_param():
    decision_process = DecisionProcess()
    v = Variable()
    p = Parameter()
    decision_process.add_helper_variables(v)
    c = CostFunction(v*p)
    with pytest.raises(NotDefinedParameterError):
        decision_process.add_cost_functions(c)


@given(build_decision_process())
@settings(suppress_health_check=(HealthCheck.too_slow,
                                 HealthCheck.large_base_example))
def test_decision_process_from_data(decision_process: DecisionProcess):
    decision_process.validate()
    decision_process_data = decision_process.get_data()
    decision_process_from_data =\
        DecisionProcess.from_data(decision_process_data)
    decision_process_from_data.validate()
    assert(decision_process_data == decision_process_from_data.get_data())


@given(build_decision_process())
@settings(suppress_health_check=(HealthCheck.too_slow,
                                 HealthCheck.large_base_example))
def test_decision_process_get_data_before_validation(
        decision_process: DecisionProcess):
    with pytest.warns(UserWarning):
        decision_process.get_data()
