from hypothesis import given, assume, settings
from hypothesis.strategies import (
    integers,
    data,
    sampled_from,
    lists,
    builds
)
from uliege.decision_process import TrajectoryOptimizer
from uliege.decision_process import (
    DecisionProcess,
    DecisionProcessRealisation,
    UselessVariableError
)
import pytest
from uliege.decision_process.datastream.datastream import Datastream
from uliege.decision_process import (
    Variable,
    VariableData,
    Parameter,
    ParameterData,
    CostFunction,
    ContainerData,
    Index,
    IndexSet,
    free_index_sets,
    sum_reduce,
    Dynamics,
    ParameterNotLengthyEnough,
    DataError
)
from typing import List, Union, Tuple
from .utils import (
    build_unconstrained_decision_process_with_datastream,
    TestDatastream,
    reasonable_lenghty_text
)
import pyomo.environ as pyo
import pyutilib.subprocess.GlobalData


pyutilib.subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False


@pytest.fixture
def solver():
    return pyo.SolverFactory("glpk")


@pytest.fixture
def fake_datastream() -> Datastream:

    class FakeDatastream(Datastream):

        def _get_initialization(self,
                                var_data: VariableData,
                                idx_seq: List[Index]) -> Union[int,
                                                               float,
                                                               None]:
            raise DataError("Not found")

        def _get_parameter(self,
                           param_data: ParameterData,
                           idx_seq: List[Index],
                           length: int) -> Union[int,
                                                 float,
                                                 None]:
            raise DataError("Not found")

        def _get_indexes_by_index_set(self,
                                      index_set: IndexSet) ->\
                Union[None, List[Index]]:
            raise DataError("Not found")

    return FakeDatastream()


@pytest.fixture
def empty_idxset_datastream() -> Datastream:

    class EmptyDatastream(Datastream):

        def _get_initialization(self,
                                var_data: VariableData,
                                idx_seq: List[Index]) -> Union[int,
                                                               float,
                                                               None]:
            return 0.0

        def _get_parameter(self,
                           param_data: ParameterData,
                           idx_seq: List[Index],
                           length: int) -> Union[int,
                                                 float,
                                                 None]:
            return [0.0]*100

        def _get_indexes_by_index_set(self,
                                      index_set: IndexSet) ->\
                Union[None, List[Index]]:
            return []

    return EmptyDatastream()


@given(build_unconstrained_decision_process_with_datastream(),
       integers(min_value=1, max_value=10))
@settings(deadline=None)
def test_controller_whole_chain_valid(
        solver,
        decision_process_with_datastream: Tuple[DecisionProcess,
                                                TestDatastream],
        T: int):
    decision_process, datastream = decision_process_with_datastream
    decision_process.validate()
    assume(T <= datastream.maximum_time_horizon)
    optim_controller = TrajectoryOptimizer(
        decision_process,
        datastream,
        time_horizon=T)
    realisation = optim_controller.solve(solver_factory=solver)
    assert(realisation.cost_sequence != dict())


@given(integers(min_value=1.0))
def test_controller_empty_decision_process_empty_solution_warn(
        fake_datastream: Datastream, T: int):
    decision_process = DecisionProcess()
    decision_process.validate()
    controller = TrajectoryOptimizer(decision_process,
                                        fake_datastream,
                                        time_horizon=T)
    with pytest.warns(UserWarning):
        realdp = controller.solve()
        empty_realdp = DecisionProcessRealisation(state_sequence=dict(),
                                                  action_sequence=dict(),
                                                  helper_sequence=dict(),
                                                  parameter_sequence=dict(),
                                                  cost_sequence=dict(),
                                                  total_cost=0)
        assert(realdp == empty_realdp)


@given(integers(min_value=1.0))
def test_controller_decision_process_not_valid_yet_warn_valid(T: int):
    decision_process = DecisionProcess()
    datastream = None
    with pytest.warns(UserWarning):
        _ = TrajectoryOptimizer(decision_process,
                                   datastream,
                                   time_horizon=T)


@given(integers(min_value=1.0))
def test_controller_decision_process_not_valid_yet_warn_invalid(T):
    decision_process = DecisionProcess()
    v = Variable()
    decision_process.add_helper_variables(v)
    decision_process.add_helper_variables(Variable())
    decision_process.add_cost_functions(CostFunction(cost_expression=v*v))
    datastream = None
    with pytest.warns(UserWarning):
        with pytest.raises(UselessVariableError):
            _ = TrajectoryOptimizer(decision_process,
                                       datastream,
                                       time_horizon=T)


@given(integers(max_value=0.0))
def test_controller_decision_process_negative_time_horizon(T: int):
    decision_process = DecisionProcess()
    datastream = None
    with pytest.raises(ValueError):
        _ = TrajectoryOptimizer(decision_process,
                                   datastream,
                                   time_horizon=T)


@given(build_unconstrained_decision_process_with_datastream(),
       integers(min_value=1, max_value=10))
def test_controller_variable_not_initializable(
        solver,
        decision_process_with_datastream: Tuple[DecisionProcess,
                                                TestDatastream],
        T: int):
    decision_process, datastream = decision_process_with_datastream
    v = Variable()
    decision_process.add_state_variables(v)
    decision_process.add_dynamics_functions(Dynamics(v, v+v))
    decision_process.add_cost_functions(CostFunction(v+v))
    decision_process.validate()
    assume(T <= datastream.maximum_time_horizon)
    optim_controller = TrajectoryOptimizer(
        decision_process,
        datastream,
        time_horizon=T)
    with pytest.raises(DataError):
        optim_controller.solve(solver)


@given(build_unconstrained_decision_process_with_datastream(),
       integers(min_value=1, max_value=10),
       data())
@settings(deadline=None)
def test_controller_parameter_no_valuation(
        solver,
        decision_process_with_datastream: Tuple[DecisionProcess,
                                                TestDatastream],
        T: int, data):
    decision_process, datastream = decision_process_with_datastream
    p = Parameter()
    v = data.draw(sampled_from(decision_process.variables))
    i_v = v[tuple([i for i in v.shape])]
    decision_process.add_parameters(p)
    expr = i_v*p
    for idx_set in free_index_sets(i_v.get_data()):
        expr = sum_reduce(expr, idx_set)

    decision_process.add_cost_functions(CostFunction(expr))
    decision_process.validate()
    assume(T <= datastream.maximum_time_horizon)
    optim_controller = TrajectoryOptimizer(
        decision_process,
        datastream,
        time_horizon=T)
    with pytest.raises(DataError):
        optim_controller.solve(solver)


@given(build_unconstrained_decision_process_with_datastream())
def test_controller_parameter_valued_but_not_lengthy_enough(
        solver,
        decision_process_with_datastream: Tuple[DecisionProcess,
                                                TestDatastream]
):
    decision_process, datastream = decision_process_with_datastream
    decision_process.validate()
    optim_controller = TrajectoryOptimizer(
        decision_process,
        datastream,
        time_horizon=200)
    with pytest.raises(ParameterNotLengthyEnough):
        optim_controller.solve(solver_factory=solver)


@given(lists(builds(IndexSet, id=reasonable_lenghty_text),
             max_size=10, min_size=1))
def test_datastream_gives_empty_sol_with_empty_set(
    solver,
    empty_idxset_datastream: Datastream,
    idx_seq: List[IndexSet]
):
    decision_process = DecisionProcess()
    v = Variable(shape=tuple(idx_seq))
    p = Parameter(shape=tuple(idx_seq))
    decision_process.add_state_variables(v)
    decision_process.add_parameters(p)
    decision_process.add_dynamics_functions(
        Dynamics(v[tuple(idx_seq)],
                 v[tuple(idx_seq)]
                 + p[tuple(idx_seq)])
    )
    expr = v[tuple(idx_seq)]*p[tuple(idx_seq)]
    for idx_set in idx_seq:
        expr = sum_reduce(expr, idx_set)
    decision_process.add_cost_functions(
        CostFunction(
            expr
        )
    )
    decision_process.validate()
    optim_controller = TrajectoryOptimizer(
        decision_process,
        empty_idxset_datastream,
        time_horizon=200)
    with pytest.warns(UserWarning):
        ctrl_realdp = optim_controller.solve(solver_factory=solver)
        empty_realdp = DecisionProcessRealisation(state_sequence=dict(),
                                                  action_sequence=dict(),
                                                  helper_sequence=dict(),
                                                  parameter_sequence=dict(),
                                                  cost_sequence=dict(),
                                                  total_cost=0)
        assert(ctrl_realdp == empty_realdp)
