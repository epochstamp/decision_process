# -*- coding: UTF-8 -*-

from typing import Union, List, Dict,  FrozenSet

from ..decision_process_components.expression.indexed_container import IndexedContainer
from ..decision_process import\
    DecisionProcess, DecisionProcessRealisation
from ..datastream.datastream import Datastream
from ..decision_process_components.variable import Variable,\
    Type, is_integer
from ..decision_process_components.parameter import Parameter
from ..decision_process_components.evaluator import Evaluator, IndexSequenceDoesNotExists
from ..decision_process_components.expression.index_set import Index,\
    IndexSet
from ..decision_process_components.cost_function import HorizonMask,\
    DiscountFactor
from ..decision_process_components.dynamics import Dynamics
from ..decision_process_components.constraint import Constraint
import pyomo.environ as pyo
from pyomo.environ import Var, Param, Set, Objective, Constraint as PyoConstraint, sum_product, value as pyovalue
import warnings
from ..utils.utils import\
    free_index_sets,\
    get_variables,\
    reduce_index_sets,\
    transform_decision_process_with_index_sets_mapping, flatten
import itertools
import numpy as np
from time import time


class TrajectoryOptimizerError(BaseException):
    pass


class DecisionProcessNotFullyKnownError(TrajectoryOptimizerError):
    pass


class FailedToFindASolution(TrajectoryOptimizerError):
    pass


class ControllerEvaluator(Evaluator):
    """Evaluator built on top of a data stream

        Attributes
        ----------
        datastream: `Datastream`
            A Controller data stream

        var_table: `dict` of `str` and `pyo.Var`
            A mapping between variable ids and pyomo variables

        param_table: `dict` of `str` and `pyo.Param`
            A mapping between parameter ids and pyomo parameters

        state_variables: `set` of `str`
            A set of variable ids that identifies variables belonging
            to the state space

        index_table: `dict` of `IndexSet` and `Index` (optional)
            A valuation of an index set by an index

        time_horizon: int
            A strictly positive integer (optional)

        t: int
            A strictly positive integer (optional)

    """

    def __init__(self,
                 datastream: Datastream,
                 var_table: Dict[str, Var],
                 param_table: Dict[str, Param],
                 state_variables: FrozenSet[str],
                 index_table: Dict[IndexSet, Index] = dict(),
                 time_horizon: int = 1,
                 t: int = 1):
        self._datastream = datastream
        self._var_table = var_table
        self._param_table = param_table
        self._index_table = index_table
        self._T = time_horizon
        self._t = t
        if (self._t > self._T):
            self._T = t
            self._t = time_horizon
        self._state_variables = state_variables
        self._shift_non_state = False
        self._param_is_set = set()
        self._index_set_memory = dict()

    def reset_param_is_set(self):
        """
            Reset the set of constructed pyomo parameters
        """
        self._param_is_set = set()

    @property
    def datastream(self):
        return self._datastream

    @datastream.setter
    def datastream(self, datastream: Datastream):
        self._datastream = datastream

    @property
    def index_table(self):
        return self._index_table

    @index_table.setter
    def index_table(self,
                    index_table: Dict[IndexSet, Index]):
        self._index_table = index_table

    @property
    def t(self):
        """
            Strictly positive time step t

            Setter behavior:

            Raises
            ---------
            ValueError
                If t is negative or strictly greater
                than time horizon
        """

        return self._t

    @t.setter
    def t(self, t: int):
        if t < 0:
            raise ValueError("t must be a non negative integer")
        elif self._t > self._T:
            err = "t must be lower than the time horizon " + str(self._T)
            raise ValueError(err)
        self._t = t

    @property
    def shift_non_state(self):
        return self._shift_non_state

    @shift_non_state.setter
    def shift_non_state(self, shift: bool):
        self._shift_non_state = shift

    def _get(self,
             container: Union[Variable, Parameter],
             idx_seq: List[Index]) -> Union[Var, Param, float, int]:
        """
            Get the value of a container, which is either
            a Pyomo variable, a Pyomo parameter
            or a scalar (for state initialization)
        """
        self._initializer = self._datastream.get_initialization
        idx_seq_i = idx_seq
        if len(idx_seq) > 1:
            for i in range(1, len(idx_seq)):
                index_set = container.shape[i]
                indexes_source = [(container.shape[j], idx_seq[j]) for j in range(i)]
                if tuple(indexes_source) + (index_set,) not in self._index_set_memory:
                    self._index_set_memory[tuple(indexes_source) + (index_set,)] = self._datastream.get_indexes_by_index_set(index_set, indexes_source)
                if idx_seq[i] not in self._index_set_memory[tuple(indexes_source) + (index_set,)]:
                    raise IndexSequenceDoesNotExists(idx_seq)

        if isinstance(container, Variable):
            if self._t > 0:
                # Inductive case with possibly a
                # time shift for non-state variables
                # Useful to model something like
                # s_{t+1} = f(s_t, a_{t+1}, ...)
                t = self._t
                if (container.id not in self._state_variables
                        and self.shift_non_state):
                    t += 1
                value = self._var_table[container.id][idx_seq_i + (t,)]
            else:
                # Basecase with possibly a
                # time shift for non-state variables
                # Useful to model something like
                # s_0 = f(s_0, a_1, ...) and
                # to initialize state variables
                if container.id in self._state_variables:
                    if (container.id+"_init", idx_seq_i) in self._param_is_set:
                        if idx_seq_i != tuple():
                            value = (self._var_table[container.id+"_init"]
                                                    [idx_seq_i])
                        else:
                            value = self._var_table[container.id+"_init"]
                    else:

                        value = self._initializer(container.get_data(), idx_seq)
                        if idx_seq_i != tuple():
                            self._var_table[container.id+"_init"][idx_seq_i].value = value
                            value = self._var_table[container.id+"_init"][idx_seq_i]
                        else:
                            self._var_table[container.id+"_init"].value = value
                            value = self._var_table[container.id+"_init"]

                        self._param_is_set.add((container.id+"_init",idx_seq_i))
                else:
                    value = (self._var_table[container.id]
                                            [idx_seq_i + (self._t + 1,)])
            errNone = "Value not found for variable " + container.id
            errNone += " indexed by " + str(idx_seq)
            errNone += " at time " + str(self._t)
            errNone += " in the current data stream."
        else:
            # Inductive and base case for parameters
            # with possibly a time shift.
            # Useful to model something like
            # s_{t+1} = f(s_t, a_t+1, p_t+1...)
            t = self._t
            if t == 0:
                t = 1
            elif self.shift_non_state:
                t += 1

            if idx_seq == tuple():
                value = self._param_table[container.id][t]
            else:
                value = self._param_table[container.id][idx_seq + (t,)]
            # value = param_seq[t]
            errNone = "Value not found for parameter " + container.id
            errNone += " indexed by " + str(idx_seq)
            errNone += " at time " + str(self._t)
            errNone += " in the current data stream."

        if value is None:
            raise ValueError(errNone)
        return value

    def _get_all_components_by_index(self,
                                     index_set: IndexSet,
                                     indexes_source: List[Index]) -> List[Index]:
        """Simply returns all the possible valuations of the index set"""
        if tuple(indexes_source) + (index_set,) not in self._index_set_memory:
            self._index_set_memory[tuple(indexes_source) + (index_set,)] = (
                self._datastream.get_indexes_by_index_set(index_set, indexes_source)
            )
        return self._index_set_memory[tuple(indexes_source) + (index_set,)]

    def _get_index(self, index_set: IndexSet) -> Index:
        """Simply returns the current valuation of the index set"""
        return self._index_table[index_set]


TYPE_CONVERT_PYOMO = {Type.REAL: pyo.Reals,
                      Type.NON_NEGATIVE_REAL: pyo.NonNegativeReals,
                      Type.NON_POSITIVE_REAL: pyo.NonPositiveReals,
                      Type.INTEGER: pyo.Integers,
                      Type.NON_NEGATIVE_INTEGER: pyo.NonNegativeIntegers,
                      Type.NON_POSITIVE_INTEGER: pyo.NonPositiveIntegers,
                      Type.BINARY: pyo.Binary}
"""dict of `Type` and `PyomoType`: Convert any Type into PyomoType
   (PyomoType is not an actual class, but Pyomo use a private class
    to inherits them all so...)
"""


def to_pyomo_type(v_type: Type):
    """


    Parameters
    ----------
    v_type: `Type`
        A variable type

    Returns
    ----------
    PyomoType
        The equivalent PyomoType (see TYPE_CONVERT_PYOMO)
    """
    return TYPE_CONVERT_PYOMO[v_type]


class TrajectoryOptimizer():

    """
        Model-based trajectory optimization.

        Provide a solution for any (in)equation-based
        deterministic decision process model.

        Attributes
        ----------
        decision_process: DecisionProcess
            A decision process model

        datastream: Datastream
            The data stream used by the controller
            to instantiate non-variable objets
            (parameters, initialization...)

        time_horizon: int (optional)
            Time horizon control. Default to 1.

        discount_factor: HorizonMask
            Discount factor of the decision process.
            Default to a DiscountFactor with 1.0 coeff (undiscounted).

        Warns
        ----------
        BaseWarning
            If the decision process has not been validated yet

        Raises
        ----------
        ValueError
            See the error message for more details

        DataError
            If any of the datastream queries have failed
            for some reason detailed in the error message

        DecisionProcessNotFullyKnownError
            If the decision process contains any function
            that is not described by an analytical expression

    """

    def __init__(self,
                 decision_process: DecisionProcess,
                 datastream: Datastream,
                 time_horizon: int = 1,
                 discount_factor: HorizonMask = DiscountFactor(1.0)):
        if not decision_process.validated:
            warn = "The decision process has not been validated yet."
            warn += " Call decision_process.validate() beforehand"
            warn += " to avoid this warning in the future."
            warn += " Current controller will attempt to validate it."
            warnings.warn(warn)
            decision_process.validate()
        if time_horizon < 1:
            raise ValueError("Time horizon should be strictly positive")
        if (not decision_process.is_fully_known()):
            err = "Some functions of your decision process"
            err += "are unknown (i.e., some functions are "
            err += "callables instead of being symbolic expressions)."
            err += "TrajectoryOptimizer does only handle fully known"
            err += "decision processes (for the moment?)."
            raise DecisionProcessNotFullyKnownError(err)
        self._decision_process = decision_process
        self._datastream = datastream
        self._time_horizon = time_horizon
        self._discount_factor = discount_factor
        self._time_horizon_sequence = range(1, time_horizon + 1)
        # Records the highest degree of the expressions
        # to decide which solver to use
        self._degree = 0
        self._model = None

    @property
    def time_horizon(self) -> int:
        return self._time_horizon

    @time_horizon.setter
    def time_horizon(self, time_horizon: int):
        if time_horizon < 1:
            raise ValueError("Time horizon should be strictly positive")
        self._time_horizon = time_horizon
        self._time_horizon_sequence = range(1, time_horizon+1)

    @property
    def datastream(self) -> Datastream:
        return self._datastream

    @datastream.setter
    def datastream(self, datastream: Datastream):
        self._datastream = datastream

    def _verify_data_stream(self) -> None:
        """ Check that no non-variable object gets non-valued.
        """
        index_sets = set()

        # Fetch index sets (free as well as part of reduce exprs)
        # over all components of the decision process
        # and verify if they evaluate to some value
        # (Even if they are empty lists)
        # Also verify that their values are subset to their
        # parent if they have one.
        index_sets = set(self._decision_process.index_sets(attached=True))
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
        for idx_set in set(self._decision_process.index_sets(attached=False)):
            if idx_set.parent is not None:
                if not set(components_idxset_getter(idx_set)).\
                        issubset(
                            set(components_idxset_getter(idx_set.parent))):
                    err = "Index set " + idx_set.id +\
                          " is not a subset of the "\
                          "index set " + idx_set.parent.id
                    raise ValueError(err)

        # Variables and parameters might not exists because
        # some index sets that are used to index them might be empty.
        # Before going on we need to transform the decision process
        # to remove these variables, as well as the expressions
        # that make use of them or that index the variables/parameters
        # with a sub index set that is empty.
        self._complete_decision_process = self._decision_process
        self._decision_process = transform_decision_process_with_index_sets_mapping(
            self._decision_process, self._index_sets_mappings
        )

        # Verify if every variables are initializable with respect
        # to their shapes
        # Also creates the index specific to each variable
        initializer = self._datastream.get_initialization
        self._variable_indexing = {}
        for v in self._decision_process.variables:
            if v.shape != tuple():
                vshape = v.shape if isinstance(v.shape, tuple) else (v.shape,)
                """
                index_comps = {idxset: components_idxset_getter(idxset, [components_idxset_getter(idx_set_mapping) for idx_set_mapping in idx_set.mapping])
                               for idxset in vshape}
                lst_index_val = (dict(zip(index_comps, x))
                                 for x in itertools.product(
                                     *index_comps.values()))
                """
                lst_index_val = self._index_sets_mappings[vshape]
                self._index_sequence_memory[tuple([idx_set.id for idx_set in vshape])] = lst_index_val
                self._variable_indexing[v.id] = Set(initialize=lst_index_val) if len(lst_index_val) > 0 else None
                if v in self._decision_process.state_variables:
                    for idx_seq in lst_index_val:
                        idx_seq = (idx_seq,) if not isinstance(idx_seq, tuple) else idx_seq
                        value = initializer(v, idx_seq)
                        if not isinstance(value, int) and not isinstance(value, float):
                            err = f"The variable {v.id}[{idx_seq}] must be initialized by a number"
                            raise TypeError(err)
                        if isinstance(value, float) and is_integer(v.v_type):
                            if not value.is_integer():
                                err = f"Variable is of type integer but the value {value} "
                                err += "is of type float"
                                raise TypeError(err)

            else:
                self._variable_indexing[v.id] = None
                if v in self._decision_process.state_variables:
                    value = initializer(v, [])
                    if not isinstance(value, int) and not isinstance(value, float):
                        err = f"The variable {v.id} must be initialized by a number"
                        raise TypeError(err)
                    if isinstance(value, float) and is_integer(v.v_type):
                        if not value.is_integer():
                            err = f"Variable {v.id} is of type integer but the value {value} "
                            err += "is of type float"
                            raise TypeError(err)

        # Verify if every parameters can be valued with respect
        # to their shapes and time horizon.
        # If parameter list length is lower than the time horizon,
        # an error is triggered.
        param_getter = self._datastream.get_parameter
        self._params_memory = dict()
        self._parameter_indexing = {}
        for p in self._decision_process.parameters:
            if p.shape != tuple():
                self._params_memory[p.id] = dict()
                pshape = p.shape if\
                isinstance(p.shape, tuple) else (p.shape,)
                lst_index_val = self._index_sets_mappings[pshape]
                self._parameter_indexing[p.id] = lst_index_val
                self._index_sequence_memory[tuple([idx_set.id for idx_set in pshape])] = lst_index_val
                for idx_seq in lst_index_val:
                    idx_seq = idx_seq if isinstance(idx_seq, tuple) else (idx_seq,)
                    self._params_memory[p.id][idx_seq] = param_getter(p.get_data(),
                                                                      idx_seq,
                                                                      length=self._time_horizon)
                
            else:
                self._params_memory[p.id] = param_getter(p.get_data(),
                                                         [],
                                                         length=self._time_horizon)
    """
    def _create_index_sets(self) -> None:
        Create Pyomo index set for each index set
            involved in the decision process
        
        c_stream = self._datastream
        for index_set in self._decision_process.index_sets():
            if self._model.component(index_set.id) is None:
                idx_seq = c_stream.get_indexes_by_index_set(index_set)
                idx_seq = [i.id for i in idx_seq]
                self._model.add_component(index_set.id,
                                          pyo.Set(initialize=idx_seq))
    """

    def _create_variables(self) -> None:
        """ Create Pyomo variables for each variable
            involved in the decision process
        """
        var_table = dict()
        variables = self._decision_process.variables
        c_stream = self._datastream
        s_ids = {v.id for v in self._decision_process.state_variables}
        h_ids = {v.id for v in self._decision_process.helper_variables}
        self._helper_activations = {}
        i = 0
        for v in variables:
            """
            # Build the index set sequence according to the variable shape
            sets = []
            vshape = v.shape if\
                isinstance(v.shape, tuple) else (v.shape,)
            for index_set in vshape:
                if self._model.component(index_set.id) is None:
                    idx_seq = c_stream.get_indexes_by_index_set(index_set)
                    idx_seq = [i.id for i in idx_seq]
                    self._model.add_component(index_set.id,
                                              pyo.Set(initialize=idx_seq))
                sets.append(self._model.component(index_set.id))
            # Add an index set for time steps
            sets.append(self._model.time_index)
            """
            # Create the pyomo variable
            vshape = v.shape if\
                isinstance(v.shape, tuple) else (v.shape,)
            var_table[v.id] = None
            time_index = self._model.time_index
            len_time_index = self._time_horizon
            if v.id in h_ids:
                self._helper_activations[v.id] = c_stream.activate_helper(v.get_data(), self._time_horizon)
                self._helper_activations[v.id] = set((np.where(self._helper_activations[v.id]))[0]+1)
                sequence_time_steps = list(self._helper_activations[v.id])
                time_index = Set(initialize=sequence_time_steps)
                len_time_index = len(sequence_time_steps)
            support = v.get_data().support
            pyo_var = None
            if len(vshape) > 0:
                if self._variable_indexing[v.id] is not None and len_time_index > 0:
                    pyo_var = Var(self._variable_indexing[v.id], time_index,
                                  name=v.id,
                                  within=to_pyomo_type(v.get_data().v_type),
                                  bounds=support)
                else:
                    continue
            else:
                if len_time_index > 0:
                    pyo_var = Var(time_index,
                                name=v.id,
                                within=to_pyomo_type(v.get_data().v_type),
                                bounds=support)
                else:
                    continue
            var_table[v.id] = pyo_var
            if v.id in s_ids:
                if len(vshape) > 0:
                    p_init = Param(self._variable_indexing[v.id],
                                   name=v.id+"_init",
                                   mutable=True)
                else:
                    p_init = Param(name=v.id+"_init",
                                   mutable=True)

                self._model.add_component(v.id+"_init", p_init)
                var_table[v.id+"_init"] = self._model.component(v.id+"_init")
            self._model.add_component(v.id, pyo_var)
        self._var_table = var_table

    def _create_parameters(self) -> None:
        """ Create Pyomo parameter for each parameter
            involved in the decision process
        """
        parameters = self._decision_process.parameters

        indexed_param_to_pyomo_param = dict()


        for p in parameters:
            # Build the index set sequence according to the parameter shape
            pshape = p.shape if\
                isinstance(p.shape, tuple) else (p.shape,)
            indexed_param_to_pyomo_param[p.id] = dict()

            if pshape == tuple():
                t = 1
                for param_value in self._params_memory[p.id]:
                    indexed_param_to_pyomo_param[p.id][t] =\
                        param_value
                    t += 1
            else:
                for idx_seq, param_values in self._params_memory[p.id].items():
                    t = 1
                    for param_value in param_values:
                        indexed_param_to_pyomo_param[p.id][idx_seq + (t,)] =\
                            param_value
                        t += 1

        self._param_table = indexed_param_to_pyomo_param

    def _prepare(self):
        """ Convert the decision process to a Pyomo model
        """
        self._helper_variables_id = {h.id for h in self._decision_process.helper_variables}
        state_variables_ids = {s.id
                               for s in self._decision_process.state_variables}
        # Build an evaluator on top of the controller datastream
        self._old_time_horizon = self._time_horizon
        self._model = pyo.ConcreteModel()
        self._model.time_index = pyo.RangeSet(1, self._time_horizon)
        self._index_sequence_memory = {}
        
        self._verify_data_stream()
        # self._create_index_sets()
        self._create_variables()
        self._create_parameters()
        self._c_eval = ControllerEvaluator(
            datastream=self._datastream,
            var_table=self._var_table,
            param_table=self._param_table,
            state_variables=state_variables_ids,
            time_horizon=self._time_horizon)
        # Set objective
        self._model.obj = Objective(rule=self._compute_objective(),
                                    sense=pyo.minimize)
        self._build_dynamics()
        self._build_constraints()

    def _compute_objective(self):
        """ Create the objective function of the decision process.
        """
        def cost_function_summation(model):
            self._c_eval.shift_non_state = False
            self._c_eval.index_table = dict()
            # Evaluate each cost function and add it up
            # according to thorizon mask and discount_factor
            
            time_sequences_per_cost_function = {
                c.id:
                    [
                        self._helper_activations[v] for v in c.variables_ids if v in self._helper_variables_id
                    ]
                for c in self._decision_process.cost_functions
            }
            self._time_sequence_per_cost_function = {
                c: (set.intersection(*tuple(flag_sequences)) if flag_sequences != [] else set(range(self._time_horizon))) for c, flag_sequences in time_sequences_per_cost_function.items()
            }

            lst_cst_func_time = [
                    (c.id, t) for c in self._decision_process.cost_functions for t in self._time_sequence_per_cost_function[c.id]
            ]
            self._model.cost_function_pyoset = pyo.Set(initialize=lst_cst_func_time)
            c_dict = {
                c.id: c for c in self._decision_process.cost_functions
            }
            self._model.timely_cost_vars = Var(self._model.cost_function_pyoset)

            def evaluate_cost_function_at_time(model, cost_function, t):
                self._c_eval.t = t
                discount_factor = self._discount_factor(t,
                                                        self._time_horizon)
                mask_factor = c_dict[cost_function].horizon_mask(t,
                                                                 self._time_horizon)                            
                if discount_factor*mask_factor != 0:
                    value = discount_factor * \
                            c_dict[cost_function](evaluator=self._c_eval,
                                                  t=t,
                                                  T=self._time_horizon)
                else:
                    value = 0

                if t == 1:
                    if hasattr(value, "polynomial_degree"):
                        self._degree = max(self._degree,
                                           value.polynomial_degree())

                return model.timely_cost_vars[cost_function, t] == value

            self._model.timely_cost_constrs = PyoConstraint(
                self._model.cost_function_pyoset,
                rule=evaluate_cost_function_at_time
            )

            total_obj = sum_product(self._model.timely_cost_vars,
                                    index=self._model.cost_function_pyoset
                                    )

            return total_obj
        return cost_function_summation

    def _dynamics_builder(self, d: Dynamics):
        """ Returns Pyomo rules for dynamics-based constraints """
        def d_builder(m, *idxs):
            # Build the index table
            idx_seq = d.state_var.get_data().idx_seq
            
            index_table = {idx_seq[i]: idxs[i]
                           for i in range(len(idx_seq))}
            t = idxs[-1]
            # Left side of the dynamics at time t (var to be updated)
            self._c_eval.index_table = index_table
            self._c_eval.t = t
            self._c_eval.shift_non_state = False
            lhs = d.state_var(evaluator=self._c_eval)

            # Right side of the dynamics at time t-1 (var update expr)
            self._c_eval.t = t - 1
            self._c_eval.shift_non_state = True
            rhs = d.state_var_update(evaluator=self._c_eval)
            # This might be seen as
            # s_{t+1} = f(s_t, a_{t+1}, p_{t+1}...)
            expr = lhs == rhs
            if t == 1:
                self._degree = max(self._degree, expr.polynomial_degree())
            return expr
        return d_builder

    def _build_dynamics(self):
        """ Build a Pyomo constraint for each dynamics.
        """
        for d in self._decision_process.dynamics_functions:
            """
            
            # Get the index set sequence of the Pyomo variable
            indexes = self._model.component(v.id).index_set()
            if None in indexes:
                indexes = []
            elif indexes.dimen == 1:
                indexes = [indexes]
            else:
                indexes = indexes.subsets()
            indexes = indexes if indexes is not None else []
            """
            # Build the Pyomo rule according to the dynamics
            sets_time_sequence = tuple([
                self._helper_activations[v] for v in d.variables_ids if v in self._helper_variables_id
            ])
            time_index = list(range(1, self._time_horizon+1))
            if len(sets_time_sequence) > 0:
                time_index = set.intersection(*sets_time_sequence)
            idx_seq = (time_index,)
            if isinstance(d.state_var, IndexedContainer):
                if self._index_sets_mappings[d.state_var.get_data().idx_seq] == []:
                    continue
                idx_seq = (self._index_sets_mappings[d.state_var.get_data().idx_seq], time_index)
            r = self._dynamics_builder(d)
            self._model.add_component(
                d.id,
                PyoConstraint(*idx_seq,
                               rule=r)
            )


    def _constraint_builder(self, c: Constraint):
        """ Returns Pyomo rules for each constraint
        """
        def c_builder(m, *idxs):
            # Get a valuation of each free index of the cosntraint
            index_table = {}
            if c.free_idx_sets != []:
                idx_seq = c.free_idx_sets[0]
                index_table = {idx_seq[i]: idxs[i]
                               for i in range(len(idx_seq))}
            t = idxs[-1]
            self._c_eval.index_table = index_table
            self._c_eval.t = t - (1 if c.shift_time_state else 0)
            self._c_eval.shift_non_state = c.shift_time_state
            # Each side of the constraint
            # refers to the same time step
            expr = c(self._c_eval)
            if t == 1:
                self._degree = max(self._degree, expr.polynomial_degree())
            return expr
        return c_builder

    def _build_constraints(self):
        """ Build a Pyomo constraint for each constraint.
        """
       
        for c in self._decision_process.constraint_functions:
            sets_time_sequence = tuple([
                self._helper_activations[v] for v in c.variables_ids if v in self._helper_variables_id
            ])
            time_index = range(1, self._time_horizon+1)
            if len(sets_time_sequence) > 0:
                time_index = set.intersection(*sets_time_sequence)
            idx_seq = (time_index,)
            free_indexes = c.free_idx_sets # Already sorted in decreasing order of length
            if free_indexes != []:
                if self._index_sets_mappings[free_indexes[0]] == []:
                    continue
                idx_seq = (self._index_sets_mappings[free_indexes[0]], time_index)
            
            r = self._constraint_builder(c)
            self._model.add_component(c.id,
                                      PyoConstraint(*idx_seq,
                                                     rule=r))

    def solve(self,
              solver_factory:
              pyo.SolverFactory = None,
              solver_verbose: bool = False)\
            -> DecisionProcessRealisation:
        """ Returns a realisation of the decision process
            following the model and the attributes
            of the controller

            (See Attributes section of this class for more details)

            Returns
            ---------
            DecisionProcessRealisation
                A realisation of this decision process control
        """
        self._prepare()
        if len(self._decision_process.variables) == 0:
            # Empty decision process calls for an empty solution
            res = DecisionProcessRealisation(state_sequence=dict(),
                                             action_sequence=dict(),
                                             helper_sequence=dict(),
                                             parameter_sequence=dict(),
                                             cost_sequence=dict(),
                                             total_cost=0.0)
            warnings.warn("Decision process is empty, hence the solution is.")
            return res
        if solver_factory is not None:
            opt = solver_factory
        else:
            max_degree = self._degree
            if max_degree <= 1:
                opt = pyo.SolverFactory("cbc")
            else:
                opt = pyo.SolverFactory("bonmin")
        results = opt.solve(self._model, tee=solver_verbose)
        if (results.solver.status != pyo.SolverStatus.ok
             or results.solver.termination_condition == pyo.TerminationCondition.infeasible
             or results.solver.termination_condition == pyo.TerminationCondition.unbounded):
            err = "Failed to solve the decision process."
            err += " This is likely because one of the constraints"
            err += " could not be satisfied or the problem is unbounded."
            err += " Details about the solver status/termination condition:"
            err += str(results.solver.status) + ", "
            err += str(results.solver.termination_condition)
            raise FailedToFindASolution(err)
        state_sequence = dict()
        for state_variable in self._complete_decision_process.state_variables:
            if state_variable.id not in self._var_table:
                state_sequence[state_variable.id] = []
        action_sequence = dict()
        for action_variable in\
                self._complete_decision_process.action_variables:
            if action_variable.id not in self._var_table:
                action_sequence[action_variable.id] = []
        helper_sequence = dict()
        for helper_variable in\
                self._complete_decision_process.helper_variables:
            if helper_variable.id not in self._var_table:
                helper_sequence[helper_variable.id] = []
        parameter_sequence = dict()
        for parameter in self._complete_decision_process.parameters:
            if parameter.id not in self._param_table:
                parameter_sequence[parameter.id] = []

        # Store sequential values in the appropriate
        # variable sequence for each variable
        for v_id, pyovar in self._var_table.items():
            is_state = False
            if v_id in {v.id for v in self._decision_process.state_variables}:
                selected_sequence = state_sequence
                is_state = True
                var = [var for var in self._decision_process.state_variables
                       if var.id == v_id][0]
            elif v_id in {v.id
                          for v in self._decision_process.action_variables}:
                selected_sequence = action_sequence
            elif v_id in {v.id
                          for v in self._decision_process.helper_variables}:
                selected_sequence = helper_sequence
            else:
                continue
            if pyovar is not None:
                offset = 1
                if is_state:
                    offset = 0
                for idx_seq_t, vardata in pyovar._data.items():
                    # Values can be either a list of numerical values
                    # or a dict for which the keys are tuple of Index objects
                    # and the values are list of numerical values
                    if isinstance(idx_seq_t, tuple):
                        if v_id not in selected_sequence:
                            selected_sequence[v_id] = dict()
                        idx_seq = tuple(idx_seq_t[:-1])
                        if idx_seq not in selected_sequence[v_id]:
                            selected_sequence[v_id][idx_seq] = [np.nan for _ in range(self._time_horizon)]
                            if is_state:
                                self._c_eval.t = 0
                                selected_sequence[v_id][idx_seq].insert(0,
                                    self._c_eval.get(var, idx_seq).value
                                )
                        value = vardata.value if vardata.value is not None else np.nan
                        selected_sequence[v_id][idx_seq][idx_seq_t[-1] - offset] = value
                    else:
                        if v_id not in selected_sequence:
                            if is_state:
                                self._c_eval.t = 0
                                selected_sequence[v_id] = [
                                    self._c_eval.get(var, tuple()).value] + [np.nan for _ in range(self._time_horizon)]
                            else:
                                selected_sequence[v_id] = [np.nan for _ in range(self._time_horizon)]
                        value = vardata.value if vardata.value is not None else np.nan
                        selected_sequence[v_id][idx_seq_t - offset] = value

        
        for helper_variable in self._decision_process.helper_variables:
            if helper_sequence.get(helper_variable.id, []) == [] and helper_variable.id in self._helper_activations and self._helper_activations.get(helper_variable.id, None) == set():
                if helper_variable.shape == tuple():
                    helper_sequence[helper_variable.id] = [np.nan] * self._time_horizon
                else:
                    helper_sequence[helper_variable.id] = dict()
                    vshape = helper_variable.shape if isinstance(helper_variable.shape, tuple) else (helper_variable.shape,)
                    for idx_seq in self._index_sets_mappings[vshape]:
                        idx_seq = (idx_seq,) if not isinstance(idx_seq, tuple) else idx_seq
                        helper_sequence[helper_variable.id][idx_seq] = [np.nan] * self._time_horizon
        # Replace nan by repeated values
        for state_id, values in state_sequence.items():
            if values != []:
                if isinstance(values, list):
                    temp_value = values[0]
                    for i in range(1, len(values)):
                        if np.isnan(values[i]):
                            state_sequence[state_id][i] = temp_value
                        else:
                            temp_value = state_sequence[state_id][i]
                else:
                    for idx_seq, values in state_sequence[state_id].items():
                        if values != []:
                            temp_value = values[0]
                            for i in range(1, len(values)):
                                if np.isnan(values[i]):
                                    state_sequence[state_id][idx_seq][i] = temp_value
                                else:
                                    temp_value = state_sequence[state_id][idx_seq][i]
        
        # Store sequential values in the
        # parameter sequence for each parameter
        parameter_sequence = self._params_memory
        obj_value = pyovalue(self._model.obj.expr)
        cost_sequence = dict()
        for cost_function in self._decision_process.cost_functions:
            if cost_function.id not in cost_sequence:
                cost_sequence[cost_function.id] = [0]*self._time_horizon
            for t in self._time_sequence_per_cost_function[cost_function.id]:
                cost_sequence[cost_function.id][t-1] = pyovalue(
                    self._model.timely_cost_vars[(cost_function.id, t)]
                )
        res = DecisionProcessRealisation(state_sequence=state_sequence,
                                         action_sequence=action_sequence,
                                         helper_sequence=helper_sequence,
                                         parameter_sequence=parameter_sequence,
                                         cost_sequence=cost_sequence,
                                         total_cost=obj_value)
        return res
