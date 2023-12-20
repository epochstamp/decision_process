from ..decision_process import DecisionProcess, DecisionProcessRealisation
from ..datastream import DatastreamFactory
from ..datastream import Datastream, DatastreamOverrider
from ..gym_env.decision_process_env import DecisionProcessEnv
from ..controllers.trajectory_optimizer import TrajectoryOptimizer
from ..controllers import ForwardController
from ..decision_process_components.expression.index_set import Index, IndexSet
from typing import Dict, Tuple, Union, List, Callable
from ..decision_process_components.variable import VariableData
from ..decision_process_components.parameter import ParameterData
import warnings
from ..decision_process_components.cost_function import DiscountFactor


class BackupActionError(BaseException):
    pass


class NoBackupActionProvided(BaseException):
    pass


class BackupActProcessor():
    """
        Directly use the user-defined backup action function
        inside the inner gym simulator
    """

    def __init__(self, backup_act_callable: Union[type(None),
                                                  Callable[[Dict,
                                                            Datastream],
                                                            Dict[str,
                                                                 Union[int,
                                                                       float,
                                                                       Dict[Tuple[IndexSet,
                                                                                  ...],
                                                                        Union[int,
                                                                              float]]]]]] = None):
        self._backup_act_callable = backup_act_callable

    def __call__(
        self,
        states: Dict[str, Union[Union[int, float],
                                Dict[Tuple[Index, ...],
                                Union[int, float]]]],
        parameters: Dict[str, Union[Union[int, float],
                                    Dict[Tuple[Index, ...],
                                    Union[int, float]]]],
        actions: Dict[str, Union[Union[int, float],
                                 Dict[Tuple[Index, ...],
                                 Union[int, float]]]],
        datastream: Datastream,
        helpers: Dict[str, Union[Union[int, float],
                                 Dict[Tuple[Index, ...],
                                 Union[int, float]]]] = dict())\
            -> Dict[str, Union[Union[int, float],
                               Dict[Tuple[Index, ...],
                               Union[int, float]]]]:
        if self._backup_act_callable is None:
            return actions
        else:
            return self._backup_act_callable(states, datastream)


class DecisionProcessSimulator():

    """
        Simulates a DecisionProcess by instantiating an environment
        and an TrajectoryOptimizer using
        two (possibly distinct) datastreams
        for each of them.

        Attributes
        ----------
        decision_process: DecisionProcess
            A decision process object
        datastream_factory: DatastreamFactory
            The datastream factory to provide datastreams to both
            the simulator and the controller


    """

    def __init__(self,
                 decision_process: DecisionProcess,
                 datastream_factory: DatastreamFactory):

        self._decision_process = decision_process
        self._datastream_factory = datastream_factory

    def simulate(self,
                 controller: ForwardController,
                 n_steps: int = 1,
                 reduce_control_horizon_towards_end: bool = False,
                 controller_time_horizon: int = 1,
                 discount_factor: float = 1,
                 solver_verbose: bool = False,
                 solver_factory=None,
                 callback: Callable=None,
                 invalid_action_fallback: Union[type(None),
                                               Callable[[Dict, Datastream],
                                                        Dict[str,
                                                             Union[int,
                                                                   float,
                                                                   Dict[Tuple[IndexSet,
                                                                              ...],
                                                                        Union[int,
                                                                              float]]]]]] = None,
                 controller_failure_fallback: Union[type(None),
                                               Callable[[Dict, Datastream],
                                                        Dict[str,
                                                             Union[int,
                                                                   float,
                                                                   Dict[Tuple[IndexSet,
                                                                              ...],
                                                                        Union[int,
                                                                              float]]]]]] = None)\
            -> DecisionProcessRealisation:
        """
            Simulate the decision process for a given number of steps
            with an TrajectoryOptimizer controller

            Parameters
            ----------
            n_steps: int (optional, default=1)
                The number of steps to simulate
            reduce_control_horizon_towards_end: bool (optional,
                                                      default = False)
                Whether the time horizon of the controller is clipped to
                the remaining number of steps
            get_sequence_of_controller_realisations: bool(optional,
                                                          default=False)
                Whether the sequence of controller
                realisations is also returned
            controller_optim_horizon: int (optional, default = 1)
                A stricty positive integer for the controller time horizon
            discount_factor: int (optional, default = 1)
                A stricty positive float provided to both
                the simulator and the controller

            Returns
            ----------
            DecisionProcessRealisation
             or tuple of DecisionProcessRealisation
                and list of DecisionProcessRealisation

                The simulation realisation and if requested through
                `get_sequence_of_controller_realisations`, the list of
                controller realisations
        """
        if controller_time_horizon < 1:
            raise ValueError(
                "Horizon optimization should be strictly positive")
        if discount_factor <= 0:
            raise ValueError("Discount factor optimization should be strictly positive")
        datastream_controller =\
            self._datastream_factory.get_controller_datastream(
                tick=0,
                control_time_horizon=controller_time_horizon
            )
        simulator = DecisionProcessEnv(self._decision_process,
                                       self._datastream_factory,
                                       time_limit=n_steps,
                                       solver_factory=solver_factory,
                                       act_processor=BackupActProcessor(
                                           invalid_action_fallback
                                        )
                                       )

        # Prepare the output of the simulator
        state_sequence = dict()
        action_sequence = dict()
        helper_sequence = dict()
        param_sequence = dict()
        cost_sequence = dict()
        total_cost = 0
        # Fetch the initial state
        initial_state = simulator.reset()
        datastream_controller = DatastreamOverrider(
            datastream_controller, initial_state
        )
        for state_id, value in initial_state.items():
            if not isinstance(value, dict):
                state_sequence[state_id] = [value]
            else:
                for idx_seq, value in initial_state[state_id].items():
                    if state_id not in state_sequence:
                        state_sequence[state_id] = dict()
                    state_sequence[state_id][idx_seq] = [value]
        controller.initialize(self._decision_process,
                              datastream_controller,
                              controller_time_horizon)
        state = initial_state
        time_horizon = controller_time_horizon
        # Start the simulation
        for i in range(n_steps):
            # Launch the controller

            
            controller.update(datastream_controller, time_horizon)
            try:
                action = controller.action(state)
            except BaseException as e:
                if controller_failure_fallback is None:
                    raise NoBackupActionProvided()
                err = "Controller failed to compute action."
                err += " Action is computed by controller fallback function."
                err += " Details: " + str(e)
                warnings.warn(err)
                try:
                    action = controller_failure_fallback(state,
                                                 datastream_controller)
                except BaseException as e2:
                    err = "Something went wrong when trying to compute"
                    err += "fallback action since the controller itself "
                    err += "failed to computes one. Details: " + str(e2) + "."
                    raise BackupActionError(err) from e

            # Simulation step
            _, reward, done, info = simulator.step(action)

            # Check whether the constraints have been violated
            # and if so terminates here with a big penalty
            if done:
                violated_constraint = info.get(
                    "constraint_violated", "notfound")
                err = "One of the constraints (id=" + \
                    violated_constraint + ") have been violated."
                err += "This means that the simulation\
                        cannot move on and that "
                err += "a big penalty have been observed."
                warnings.warn(err)
                return DecisionProcessRealisation(
                    state_sequence=state_sequence,
                    action_sequence=action_sequence,
                    helper_sequence=helper_sequence,
                    parameter_sequence=param_sequence,
                    cost_sequence=cost_sequence,
                    total_cost=total_cost
                )

            # Fill the output with the current action
            for action, value in info["action"].items():
                if not isinstance(value, dict):
                    action_sequence[action] = action_sequence.get(
                        action, []) + [value]
                else:
                    for idx_seq, value in value.items():
                        if action not in action_sequence:
                            action_sequence[action] = dict()
                        action_sequence[action][idx_seq] =\
                            action_sequence[action].get(idx_seq, []) + [value]

            # Fill the output with the next state
            for state, value in info["state"].items():
                if not isinstance(value, dict):
                    v = value[-1] if isinstance(value, list) else value
                    state_sequence[state] = state_sequence.get(
                        state, []) + [float(v)]
                else:
                    for idx_seq, value in value.items():
                        v = value[-1] if isinstance(value, list) else value
                        if state not in state_sequence:
                            state_sequence[state] = dict()
                        state_sequence[state][idx_seq] =\
                            state_sequence[state].get(idx_seq, []) + [float(v)]

            # Fill the output with the helper variables
            for helper, value in info["helper"].items():
                if not isinstance(value, dict):
                    v = value[-1] if isinstance(value, list) else value
                    helper_sequence[helper] = helper_sequence.get(
                        helper, []) + [float(v)]
                else:
                    for idx_seq, value in value.items():
                        v = value[-1] if isinstance(value, list) else value
                        if helper not in helper_sequence:
                            helper_sequence[helper] = dict()
                        helper_sequence[helper][idx_seq] =\
                            helper_sequence[helper].get(idx_seq, [])\
                            + [float(v)]

            # Fill the output with the current parameters
            for param, value in info["param"].items():
                if not isinstance(value, dict):
                    v = value[-1] if isinstance(value, list) else value
                    param_sequence[param] = param_sequence.get(
                        param, []) + [float(v)]
                else:
                    for idx_seq, value in value.items():
                        v = value[-1] if isinstance(value, list) else value
                        if param not in param_sequence:
                            param_sequence[param] = dict()
                        param_sequence[param][idx_seq] =\
                            param_sequence[param].get(idx_seq, []) + [float(v)]

            # Fill the output with the cost functions collected from the
            # simulator step
            for cost, value in info["cost"].items():
                if cost not in cost_sequence:
                    cost_sequence[cost] = list()
                cost_sequence[cost].append(value)
            total_cost += (discount_factor**i)*reward

            # Update the datastream for the controller
            # (this is already done for the simulator side inside the gym env)
            if reduce_control_horizon_towards_end:
                time_horizon = max(min(controller_time_horizon, n_steps - (i + 1)), 1)
            else:
                time_horizon = controller_time_horizon
            datastream_controller =\
                self._datastream_factory.get_controller_datastream(i + 1, time_horizon)
            datastream_controller = DatastreamOverrider(
                datastream_controller, info["state"])
            state = info["state"]
            if callback is not None:
                info_callback = dict(info)
                info_callback["current_time_step"] = i
                info_callback["progress"] = i/n_steps
                callback(info_callback)
            

        # Returns the simulator realisation
        return DecisionProcessRealisation(
            state_sequence=state_sequence,
            action_sequence=action_sequence,
            helper_sequence=helper_sequence,
            parameter_sequence=param_sequence,
            cost_sequence=cost_sequence,
            total_cost=total_cost
        )
