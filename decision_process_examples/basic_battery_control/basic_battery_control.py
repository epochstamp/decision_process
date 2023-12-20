# -*- coding: UTF-8 -*-
from uliege.decision_process import Dynamics, Datastream
from uliege.decision_process import Type, Variable
from uliege.decision_process import Parameter
from uliege.decision_process import ForwardController
from uliege.decision_process import (
    CostFunction,
    LastTimeStepMask
)
from uliege.decision_process.datastream import DatastreamOverrider
from uliege.decision_process import Constraint
from uliege.decision_process import DecisionProcess, DecisionProcessError
from uliege.decision_process import TrajectoryOptimizer
from uliege.decision_process import OptimForwardController
from uliege.decision_process import sum_reduce
from uliege.decision_process import IndexSet
import pprint
from scripts.decision_process_to_pdf import decision_process_to_pdf
from decision_process_examples.\
    basic_battery_control.basic_battery_stream import\
    BasicBatteryStream
from uliege.decision_process import DatastreamFactory
from uliege.decision_process import DecisionProcessSimulator
import pyomo.environ as pyo
import itertools
import logging
from time import time
logging.getLogger('pyomo.core').setLevel(logging.ERROR)


class StrikeController(ForwardController):
    """
    A controller that refuses to do control
    """
    def initialize(self,
                   decision_process: DecisionProcess,
                   datastream: Datastream,
                   control_time_horizon: int = 1):
        pass

    def update(self,
               datastream: Datastream,
               control_time_horizon: int = 1):
        pass

    def action(self, state):
        raise BaseException("Deal with it yourself :)")


class BigActionsController(ForwardController):

    def initialize(self,
                   decision_process: DecisionProcess,
                   datastream: Datastream,
                   control_time_horizon: int = 1):
        self._decision_process = decision_process
        self._datastream = datastream
        self._control_time_horizon = control_time_horizon

    def update(self,
               datastream: Datastream,
               control_time_horizon: int = 1):
        self._datastream = datastream
        self._control_time_horizon = control_time_horizon

    def action(self, state):
        d_actions = dict()
        big_value = 1000000000
        for action in self._decision_process.action_variables:
            if action.shape == tuple():
                d_actions[action.id] = big_value
            else:
                d_actions[action.id] = dict()
                action_shape = action.shape
                if isinstance(action_shape, IndexSet):
                    action_shape = (action_shape,)
                index_comps = {idxset: self._datastream.
                                get_indexes_by_index_set(idxset)
                                for idxset in action_shape}
                lst_index_val = (dict(zip(index_comps, x))
                                    for x in itertools.product(
                                    *index_comps.values()))
                for idx_mapping in lst_index_val:
                    idx_seq = tuple([idx_mapping[i] for i in action_shape])
                    d_actions[action.id][idx_seq] = big_value
        return d_actions


def create_decision_process():

    """
    The decision process

    This is a dummy storage control problem.
    The goal is to charge as most as possible at the control end,
    with a cumulative charge constraints for all storages.
    """
    decision_process =\
        DecisionProcess(
            id="Basic battery decision process",
            description="Simple battery problem which consists\
                         in charging as much as possible all the batteries.")

    # Index sets

    # linear_batteries is a subset of 'batteries'
    # TODO : Id instead of set_name ? mandatory ?
    batteries = IndexSet(id="batteries", description="Set of batteries")
    linear_batteries = IndexSet(
        id="linear_batteries",
        parent=batteries,
        description="Subset of batteries with linear dynamics"
    )

    # Parameters

    # Maximum charge power
    max_charge_pow = Parameter(id="max_charge_power", shape=(batteries,))
    # Maximum discharge power
    max_discharge_pow = Parameter(id="max_discharge_power", shape=(batteries,))
    # Maximum total cumulative charge power
    max_cum_charge = Parameter(id="max_cumulative_power", shape=tuple())
    # Delta time
    dt = Parameter(id="delta_time", shape=tuple())
    # Charge efficiency
    charge_efficiency = Parameter(id="charge_efficiency", shape=(batteries,))
    # Discharge efficiency
    discharge_efficiency = Parameter(id="discharge_efficiency",
                                     shape=(batteries,))
    # State of charge valorisation
    state_of_charge_val = Parameter(id="state_of_charge_val",
                                    shape=(batteries,))

    # Parameter space
    parameters = (max_charge_pow,
                  max_discharge_pow,
                  max_cum_charge,
                  dt,
                  charge_efficiency,
                  discharge_efficiency,
                  state_of_charge_val)

    # Add all parameters to the decision process
    decision_process.add_parameters(*parameters)

    # State of charge each battery
    state_of_charge = Variable(id="state_of_charge",
                               v_type=Type.NON_NEGATIVE_REAL,
                               shape=(batteries,))

    # Cumulative charge power applied to all batteries
    cumulative_charge = Variable(id="cumulative_charge",
                                 v_type=Type.NON_NEGATIVE_REAL)

    # State space
    state_variables = (state_of_charge, cumulative_charge)

    # Add all state variables to the decision process
    decision_process.add_state_variables(*state_variables)

    # Charge power applied to each battery
    charge_power = Variable(id="charge_power",
                            v_type=Type.NON_NEGATIVE_REAL,
                            shape=(batteries,))

    # Discharge power applied to each battery
    discharge_power = Variable(id="discharge_power",
                               v_type=Type.NON_NEGATIVE_REAL,
                               shape=(batteries,))

    # Action space
    action_variables = (charge_power, discharge_power)

    # Add all action variables to the decision process
    decision_process.add_action_variables(*action_variables)

    # Dynamics : state of charge update

    state_of_charge_update =\
        state_of_charge[linear_batteries]\
        + (charge_efficiency[linear_batteries]
           * charge_power[linear_batteries])\
        * dt\
        - (discharge_power[linear_batteries]
           / discharge_efficiency[linear_batteries])\
        * dt

    dyns_1 = Dynamics(id="linear_state_of_charge_update",
                      state_var=state_of_charge[linear_batteries],
                      state_var_update=state_of_charge_update)

    # Dynamics : cumulative charge
    cumulative_charge_inner_expr = charge_efficiency[batteries] *\
        charge_power[batteries]
    cumulative_charge_sum = sum_reduce(inner_expr=cumulative_charge_inner_expr,
                                       idx_set=batteries)

    cumulative_charge_update =\
        cumulative_charge\
        + cumulative_charge_sum\
        * dt

    dyns_2 = Dynamics(id="cumulative charge update",
                      state_var=cumulative_charge,
                      state_var_update=cumulative_charge_update)

    # Dynamics set
    batteries_dynamics = (dyns_1, dyns_2)

    # Add all dynamics to the decision process
    decision_process.add_dynamics_functions(*batteries_dynamics)
    # Constraint 1 : Maximum charge power
    max_charge_expr = charge_power[batteries] <= max_charge_pow[batteries]
    max_charge_constr = Constraint(id="max_charge_batteries",
                                   ineq=max_charge_expr)

    # Constraint 2 : Maximum discharge power
    max_discharge_expr = discharge_power[batteries] <=\
        max_discharge_pow[batteries]
    max_discharge_constr = Constraint(id="max_discharge_batteries",
                                      ineq=max_discharge_expr)

    # Constraint 3 : Maximum cumulative charge
    max_cumulative_charge_expr = cumulative_charge <= max_cum_charge
    max_cumulative_charge_constr = Constraint(id="max_cumulative_charge",
                                              ineq=max_cumulative_charge_expr)

    # Constraint set
    batteries_constraints = (max_charge_constr,
                             max_discharge_constr,
                             max_cumulative_charge_constr)

    # Add all constraints to the decision process
    decision_process.add_constraint_functions(*batteries_constraints)

    # Cost function 1 : Content of the battery
    soc_sum = state_of_charge_val[batteries] *\
        state_of_charge[batteries]
    state_of_charge_reward = sum_reduce(inner_expr=soc_sum,
                                        idx_set=batteries)
    battery_cost = CostFunction(id="total_state_of_charge",
                                cost_expression=-state_of_charge_reward,
                                horizon_mask=LastTimeStepMask())

    batteries_cost_functions = {battery_cost}

    # Add all constraints to the decision process
    decision_process.add_cost_functions(*batteries_cost_functions)

    return decision_process


class BasicDataStreamFactory(DatastreamFactory):

    def _get_controller_datastream(self, tick: int = 0, control_time_horizon: int = 1):
        return BasicBatteryStream(tick)

    def _get_simulation_datastream(self, tick: int = 0):
        return BasicBatteryStream(tick)


if __name__ == "__main__":
    decision_process = create_decision_process()
    try:
        decision_process.validate()
        print("Decision process has been successfully validated. Display: ")
    except DecisionProcessError as e:
        print("An error occured during decision process validation")
        print(e)
        exit()

    pprint.pprint(decision_process.get_data().dict())

    c_datastream = BasicBatteryStream()
    controller = TrajectoryOptimizer(decision_process=decision_process,
                                     datastream=c_datastream,
                                     time_horizon=10000)
    print ("Time took for the whole realisation (seconds):")
    t = time()
    realisation = controller.solve()
    print (time() - t)
    exit()
    pprint.pprint(realisation.dict())
    cont_id_mapping = {"cumulative_charge": "cc",
                       "delta_time": r"\delta",
                       "charge_power": "cp",
                       "discharge_power": "dp",
                       "state_of_charge": "soc",
                       "charge_efficiency": "ce",
                       "discharge_efficiency": "de",
                       "max_charge_power": "max_cp",
                       "max_cumulative_power": "max_cc",
                       "max_discharge_power": "max_dp",
                       "state_of_charge_val": "soc_val"}
    func_id_mapping = {"total_state_of_charge": "total_soc"}
    idxset_id_mapping = {"batteries": "B",
                         "linear_batteries": "LinB"}
    decision_process_to_pdf(decision_process.get_data(),
                            idxset_id_mapping=idxset_id_mapping,
                            cont_id_mapping=cont_id_mapping,
                            func_id_mapping=func_id_mapping,
                            get_tex=True)

    simulator = DecisionProcessSimulator(
                    decision_process,
                    BasicDataStreamFactory()
                )

    trajectory_optimizer_controller =\
        OptimForwardController(time_horizon=1000000,
                               solver_factory=pyo.SolverFactory("cplex"))

    trajectory_optimizer_controller.initialize(
        decision_process,
        c_datastream,
        10
    )


    print ("Time took for the whole simulation (seconds):")
    t = time()
    simulated =\
        simulator.simulate(controller=trajectory_optimizer_controller,
                           n_steps=1,
                           reduce_control_horizon_towards_end=True,
                           controller_time_horizon=1000000)\

    print (time() - t)

    print(realisation)
    print(simulated)
    if (realisation == simulated):
        print("As expected, realisation and simulation results\
               do totally match")
    else:
        print("Something weird happened, realisation and simulation results\
               should match")

