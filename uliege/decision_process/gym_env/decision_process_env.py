import gym
from ..decision_process import DecisionProcess
from ..decision_process_components.expression.index_set import Index, IndexSet
from ..datastream.datastream import Datastream, OutOfBoundsVarValueError
from gym.spaces import Box, Discrete, Space, Dict as SpaceDict
import numpy as np
from typing import Dict, List, Tuple, Union
import warnings
from ..decision_process_components.evaluator import Evaluator
from ..decision_process_components.variable import\
    Variable, is_real, is_integer, VariableData
from ..decision_process_components.parameter import Parameter, ParameterData
from ..gym_env.drawer import Drawer, NoDrawer
from ..gym_env.obs_processor import ObsProcessor, StateSpace
from ..gym_env.gym_evaluator import GymEvaluator
import itertools
from ..gym_env.util import get_dict_space_from_container_set
from ..decision_process_components.expression.reduce import sum_reduce
from ..decision_process_components.dynamics import Dynamics
from ..decision_process_components.cost_function import CostFunction
from ..utils.utils import flatten, free_index_sets
from ..decision_process_components.constraint import Constraint
from copy import deepcopy
from ..datastream.datastream_factory import DatastreamFactory
from ..decision_process_components.constraint import Constraint
from ..gym_env.act_processor import ActProcessor, ActionSpace
from ..utils.utils import transform_decision_process_with_index_sets_mapping
from ..controllers.trajectory_optimizer import\
    TrajectoryOptimizer, FailedToFindASolution


class EmptyTrajectoriesError():
    pass


class ShorterTrajectoriesError():
    pass


class DatastreamOverrider(Datastream):

    """
        Overrides the initial state
        and some parameters
        (useful for simulating a step with pre-fixed actions)

        Attributes
        ----------
        datastream: Datastream
            a Datastream object
        initial_state: dict of str to float/int
                       or dict of tuple of index to float/int
            Values for initial (possibly indexed) states
        parameters: dict of str to float/int
                       or dict of tuple of index to float/int
            Values for initial (possibly indexed) parameters
    """

    def __init__(self,
                 datastream: Datastream,
                 initial_state: Dict[str,
                                     Union[float,
                                           int,
                                           Dict[Tuple[Index, ...],
                                                Union[float,
                                                      int]]]],
                 parameters: Dict[str,
                                  Union[float,
                                        int,
                                        Dict[Tuple[Index, ...],
                                             Union[float,
                                                   int]]]]):
        self._datastream = datastream
        self._initial_state = initial_state
        self._parameters = parameters

    def _get_parameter(self,
                       param_data: ParameterData,
                       idx_seq: List[Index],
                       length: int):
        """
            Get the (possibly indexed) parameter from either
            the current or the overrided datastream
        """
        if param_data.id in self._parameters:
            if len(idx_seq) == 0:
                return self._parameters[param_data.id]
            else:
                return self._parameters[param_data.id][tuple(idx_seq)]
        return self._datastream.get_parameter(param_data, idx_seq, 1)

    def _get_indexes_by_index_set(self,
                                  index_set: IndexSet,
                                  indexes_source: List[Tuple[IndexSet, Index]] = []) -> Union[None,
                                                                List[Index]]:
        """
            Get the (possibly empty) set of indexes of an index set
            from the datastream
        """
        return self._datastream.get_indexes_by_index_set(index_set, indexes_source)

    def activate_helper(self, var_data: VariableData, T: int) -> List[bool]:
        """
            Returns a boolean vector of length T which indicates whether a helper variable should be created each discrete time step between 0 and T-1
        """
        return self._datastream.activate_helper(var_data, T)

    def _get_initialization(self,
                            var_data: VariableData,
                            idx_seq: List[Index]):
        """
            Get the overrided initial state
        """
        if len(idx_seq) == 0:
            value = self._initial_state[var_data.id]
        else:
            idx_seq = tuple(idx_seq)
            value = self._initial_state[var_data.id][idx_seq]
        return value[-1] if isinstance(value, list) else value


class DecisionProcessEnv(gym.Env):

    """
        Implementation of the gym interface
        to simulate a DecisionProcess

        Attributes
        ----------
        decision_process: DecisionProcess
            A decision process
        datastream_factory: DatastreamFactory
            A DatastreamFactory object (to update datastream at each step)
        drawer: Drawer
            A Drawer object to render the decision process's state
        obs_processor: ObsProcessor
            An ObsProcessor object
            (postprocess the state of the decision process
            for e.g., deep reinforcement learning algorithms)
                (optional, default=StateSpace())
        act_processor: ActProcessor
            An ActProcessor object to modify actions
            when they break the constraints
                (optional, default=ActionSpace())
        penalty_constraint_break: int (optional, default=0)
            Penalty (strictly positive, is negated when returned)
            to apply when the constraints are broken
            despite the postprocessing (optional, default to 100000)
        time_limit: int (optional, default=100)
            The number of steps the simulator is expected to roll
    """

    def __init__(self,
                 decision_process: DecisionProcess,
                 datastream_factory: DatastreamFactory,
                 drawer: Drawer = NoDrawer(),
                 obs_processor: ObsProcessor = StateSpace(),
                 act_processor: ActProcessor = ActionSpace(),
                 penalty_constraint_break=100000,
                 time_limit: int = 100,
                 solver_factory=None):
        if time_limit < 1:
            raise ValueError("Time limit should be strictly positive")
        self._time_limit = time_limit
        self._complete_decision_process = decision_process
        self._datastream_factory = datastream_factory
        self._datastream = datastream_factory.get_simulation_datastream()

        index_sets = set(decision_process.index_sets(attached=True))
        components_idxset_getter = self._datastream.get_indexes_by_index_set
        self._index_sets_mappings = {}
        for index_set_sequence in index_sets:
            if isinstance(index_set_sequence, IndexSet):
                index_set_sequence = [index_set_sequence]
            lst_index_val = list(components_idxset_getter(index_set_sequence[0]))
            i = 1
            for idx_set in index_set_sequence[1:]:
                lst_index_val = [tuple(i) if isinstance(i, tuple) else (i,) for i in lst_index_val]
                lst_mapping = list(index_set_sequence[:i])
                lst_index_val = flatten(
                    [
                        [
                            tuple(index_val) + tuple((index_val_target,)) for index_val_target in components_idxset_getter(idx_set, [(lst_mapping[j],index_val[j]) for j in range(i)])
                        ] for index_val in lst_index_val
                    ]
                )
                i += 1
            self._index_sets_mappings[tuple(index_set_sequence)] = lst_index_val

        self._decision_process =\
            transform_decision_process_with_index_sets_mapping(
                self._complete_decision_process,
                self._index_sets_mappings
            )
        self._drawer = drawer
        self._obs_processor = obs_processor
        self.observation_space = obs_processor.get_space(
            decision_process,
            self._datastream.get_indexes_by_index_set
        )
        self.action_variables = get_dict_space_from_container_set(
            decision_process.action_variables,
            self._datastream.get_indexes_by_index_set)
        self._state = None
        self._act_processor = act_processor
        if penalty_constraint_break < 0:
            err = "Penalty should be strictly positive"
            err += " to be negated in the reward return"
            raise ValueError(err)
        self._penalty_constraint_break = penalty_constraint_break
        self._observation = None
        self._surrogate_decision_process = None
        self._solver_factory = solver_factory

    def _create_surrogate_controller(self):
        """
            Create the constrained controller to execute an action
            in the decision process at current state
        """
        self._surrogate_decision_process = deepcopy(self._decision_process)
        # Put constraints on the actions
        cstrs = []
        params = []
        for action_variable in\
                self._surrogate_decision_process.action_variables:
            shape = action_variable.shape
            value = Parameter(id="affects_" + action_variable.id,
                              shape=shape)
            params.append(value)
            action_shape = action_variable.shape
            if not isinstance(action_shape, tuple):
                action_shape = (action_shape,)
            if len(action_shape) == 0:
                cstrs.append(Constraint(
                    action_variable == value
                ))
            else:
                cstrs.append(Constraint(
                    action_variable[action_shape] == value[action_shape]
                ))
        self._surrogate_decision_process.add_parameters(*params)
        self._surrogate_decision_process.add_constraint_functions(*cstrs)
        self._surrogate_decision_process.validate()
        self._updater =\
            TrajectoryOptimizer(self._surrogate_decision_process,
                                   self._datastream)

    def _apply_action(self, action: Dict[str,
                                         Union[Union[float, int],
                                               Dict[Tuple[Index, ...],
                                                    Union[float,
                                                          int]]]]):
        """
            Apply an action to the current state of the decision process
        """
        param_valuation =\
            {"affects_"+a.id: action[a.id]
                for a in self._surrogate_decision_process.action_variables}
        self._surrogate_datastream = \
            DatastreamOverrider(
                self._datastream,
                self._state,
                param_valuation
            )
        self._updater.datastream = self._surrogate_datastream
        return self._updater.solve(solver_factory=self._solver_factory)

    def step(self, action: Dict[str,
                                Union[Union[float, int],
                                      Dict[Tuple[Index, ...],
                                           Union[float,
                                                 int]]]])\
            -> Tuple[Dict, float, bool, Dict]:
        """
            Make a step in the decision process simulation

            Parameters
            ----------
            action: dict of str to int/float or tuple of Index to int/float
                The action to perform to make a step in the simulation

            Returns
            ----------
            tuple of dict, float, bool, dict
                The first member is the current observation after the step.

                The second member is the sum of cost functions at current time.

                The third member is true if the state is terminal.
                    (i.e. if the constraints are broken)

                The fourth member is a dict that contains additionnal
                    informations that should be useful to the object
                    using this interface to actually perform a simulation
        """
        total_cost = None
        info = dict()
        post_action = action
        if self._decision_process.is_fully_known():
            # Create a surrogate decision process where actions and states are
            # constrained to be equal to some value
            try:
                result = self._apply_action(action)
            except FailedToFindASolution as e1:
                try:
                    post_action =\
                        self._act_processor(self._state,
                                            self._params,
                                            action,
                                            self._datastream)
                    result = self._apply_action(post_action)
                except FailedToFindASolution as e2:
                    warning = "Failed to step forward in the environment\
                              with the current action,"
                    warning = " despite the user-defined\
                                action compatibility postprocessor."
                    warning += "This is likely because\
                                one of the constraints is broken."
                    warning += "See 'info' dict for more details."
                    info["error_before_postaction"] = e1
                    info["error_after_postaction"] = e2
                    info["state"] = self._state
                    info["action"] = action
                    info["postaction"] = post_action
                    info["param"] = self._params
                    info["costs"] =\
                        {c.id: -self._penalty_constraint_break
                         for c in self._decision_process.cost_functions}
                    warnings.warn(warning)
                    return (self._observation,
                            -self._penalty_constraint_break,
                            True,
                            info)

            # State has been successfully updated
            self._state = result.state_sequence
            # By convention, if the next state is NaN, we consider that the value never changed
            self._observation = self._obs_processor(self._state,
                                                    self._params,
                                                    self._datastream)
            info["action"] = post_action
            info["state"] = self._state
            info["param"] = result.parameter_sequence
            info["helper"] = result.helper_sequence
            info["cost"] =\
                {c.id: (0 if (c.horizon_mask(self._tick+1,
                                            self._time_limit) == 0 or result.cost_sequence[c.id] == [])
                        else ((result.cost_sequence[c.id][0]
                              / c.horizon_mask(self._tick + 1,
                                               self._tick + 1))
                              * c.horizon_mask(self._tick + 1,
                                               self._time_limit)))
                    for c in self._decision_process.cost_functions}
            total_cost = sum(info["cost"].values())
        else:
            # IDEA : use scipy optimize?
            err = "For this moment only fully known "
            err += "decision processes are handled in this env."
            err += " We are aware that handling non-fully known"
            err += " decision processes is a desirable feature"
            err += " and we are currently working on."
            raise NotImplementedError(err)
        self._tick = (self._tick + 1) % self._time_limit
        self._datastream =\
            self._datastream_factory.get_simulation_datastream(self._tick)
        self._update_params_dict()
        return self._observation, total_cost, False, info

    def _update_params_dict(self):
        """
            Updates the internal parameters dict
            using the current datastream
        """
        for p in self._decision_process.parameters:
            pshape = p.shape if\
                isinstance(p.shape, tuple) else (p.shape,) 
            if p.id not in self._params:
                if pshape != tuple():
                    self._params[p.id] = dict()
            if pshape == tuple():
                self._params[p.id] = self._datastream.\
                    get_parameter(p.get_data(), tuple(), 1)
            else:
                lst_index_val = self._index_sets_mappings[pshape]


                for idx_seq in lst_index_val:
                    if not isinstance(idx_seq, tuple):
                        idx_seq = (idx_seq,)
                    self._params[p.id][idx_seq] = self._datastream.\
                        get_parameter(p.get_data(), idx_seq, 1)

    def reset(self) -> Dict:
        """
            Reset the simulation to the original initial state

            Returns
            ----------
            dict
                A dict describing the environment observation
                (user-defined, see __init__)
        """
        self._tick = 0
        self._datastream =\
            self._datastream_factory.get_simulation_datastream(0)
        if self._state is None:
            self._state = dict()
        for v in self._decision_process.state_variables:
            vshape = v.shape if\
                isinstance(v.shape, tuple) else (v.shape,) 
            if v.id not in self._state:
                if vshape != tuple():
                    self._state[v.id] = dict()
            if vshape == tuple():
                self._state[v.id] = self._datastream.\
                    get_initialization(v.get_data(), tuple())
            else:
                components_by_index_set_func =\
                    self._datastream.get_indexes_by_index_set
                lst_index_val = list(components_by_index_set_func(vshape[0]))
                i = 1
                for idx_set in vshape[1:]:
                    lst_index_val = [tuple(i) if isinstance(i, tuple) else (i,) for i in lst_index_val]
                    lst_mapping = list(vshape[:i])
                    lst_index_val = flatten(
                        [
                            [
                                tuple(index_val) + tuple((index_val_target,)) for index_val_target in components_by_index_set_func(idx_set, [(lst_mapping[j],index_val[j]) for j in range(i)])
                            ] for index_val in lst_index_val
                        ]
                    )
                    i += 1
                for idx_seq in lst_index_val:
                    if not isinstance(idx_seq, tuple):
                        idx_seq = (idx_seq,)
                    self._state[v.id][idx_seq] = self._datastream.\
                        get_initialization(v.get_data(), idx_seq)
        self._actions = dict()
        self._params = dict()
        self._update_params_dict()
        self._observation = self._obs_processor(self._state,
                                                self._params,
                                                self._datastream)
        self._create_surrogate_controller()
        return self._observation

    def render(self, mode='human', close=False) -> None:
        """
            Renders the current state of the decision process
            (user-defined, see __init__)

            Parameters
            ----------
                mode: str (optional, default='human')
                    Not used
                close: bool (optional, default=False)
                    Close the drawer object
                    (usually called at the end of the simulation)
        """
        if self._observation is not None:
            self._drawer.draw(self._state, self._params, self._actions,
                              self._datastream)
        else:
            warning = "No observation available"
            warning = " at this moment, thus nothing will be drawn."
            warning = "Consider calling 'reset()' to this env."
            warnings.warn(warning)
        if close:
            self._drawer.close()
