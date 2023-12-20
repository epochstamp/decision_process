# -*- coding: UTF-8 -*-
from __future__ import annotations
from .decision_process_components.dynamics import Dynamics,\
                                                 DynamicsData
from .decision_process_components.constraint import Constraint,\
                                                   ConstraintData
from .decision_process_components.cost_function import CostFunction,\
                                                      CostFunctionData
from .base.base_component import BaseComponent, BaseComponentData
from typing import Dict, List, Tuple, Union, FrozenSet
from pydantic import BaseModel
from .decision_process_components.expression.index_set import Index, IndexSet
from .decision_process_components.expression.expression import Expression
from .decision_process_components.parameter import Parameter,\
                                                  ParameterData
from .decision_process_components.variable import Variable,\
                                                 VariableData
from .utils.utils import get_variables, get_parameters,\
    SortedAndEqualedSetByKey as sorted_set
import warnings
from sortedcontainers import SortedKeyList as sorted_key_list


def sorted_by_id(o):
    return o.id


"""
    A ContainerSequence is a dict that map strings to either
        - lists of float or lists of integers
        - dicts that maps lists of Index to list of float
          or list of integers
"""
ContainerSequence = Dict[str,
                         Union[List[float], List[int],
                               Dict[Tuple[Index, ...],
                                    Union[List[float],
                                          List[int]]]]]


class DecisionProcessRealisation(BaseModel):
    """ Decision process sequential realisation.

        Useful to represent a decision process
        over time following a policy
        computed by a controller.

        ContainerSequence is a dict object
        which maps str keys to dicts
        which map a tuple of indexes to a list
        of numeric values.

        Attributes
        ----------
        state_sequence: ContainerSequence
            Realisation of the state variables over time
        action_sequence: ContainerSequence
            Realisation of the action variables over time
        helper_sequence: ContainerSequence
            Realisation of the helper variables over time
        parameter_sequence: ContainerSequence
            Realisation of the parameters over time
        total_cost: float
            Total cost computed from the realisation
            according to the decision process
    """

    state_sequence: ContainerSequence
    action_sequence: ContainerSequence
    helper_sequence: ContainerSequence
    parameter_sequence: ContainerSequence
    cost_sequence: Dict[str, Union[List[float], List[int]]]
    total_cost: float


class DecisionProcessData(BaseComponentData):
    """ `DecisionProcess` DTO

         Attributes
         ----------
         state_variables: list of Variable
            State variables
         action_variables: list of Variable
            Action variables
         helper_variables: list of Variable
            Helper variables
         parameters: list of Parameter
            Parameter space
         dynamics_functions: list of Dynamics
            Dynamics space
         constraint_functions: list of Constraints
            Constraint space
         cost_functions: list of CostFunction
            Cost function space
    """
    state_variables: List[VariableData]
    action_variables: List[VariableData]
    parameters: List[ParameterData]
    dynamics_functions: List[DynamicsData]
    constraint_functions: List[ConstraintData]
    cost_functions: List[CostFunctionData]
    helper_variables: List[VariableData] = set()


class DecisionProcessError(BaseException):
    pass


class AlreadyExistsError(DecisionProcessError):
    pass


class NotAStateVariableError(DecisionProcessError):
    pass


class NotDefinedVariableError(DecisionProcessError):
    pass


class NotDefinedParameterError(DecisionProcessError):
    pass


class UselessVariableError(DecisionProcessError):
    pass


class UselessParameterError(DecisionProcessError):
    pass


class DynamicsLessStateVariableError(DecisionProcessError):
    pass


class NoCostFunctionError(DecisionProcessError):
    pass


class DecisionProcess(BaseComponent):
    """ Model-based decision process



        Attributes
        ----------
        id: str (optional)
            The id of the decision process



    """

    def __init__(self,
                 id: str = "",
                 description: str = ""):
        super().__init__(id=id, description=description)
        self._validated = False
        self._state_variables = sorted_set(key=sorted_by_id)
        self._action_variables = sorted_set(key=sorted_by_id)
        self._helper_variables = sorted_set(key=sorted_by_id)
        self._parameters = sorted_set(key=sorted_by_id)
        self._variables = sorted_set(key=sorted_by_id)
        self._dynamics_functions = set()
        self._constraint_functions = set()
        self._cost_functions = set()
        self._functions = set()
        self._index_sets = None

    @property
    def validated(self):
        """ Whether the model is validated
            (Mandatory for using it in controllers)
        """
        return self._validated

    @property
    def variables(self):
        """ Variable space of the decision process
        """
        return self._variables

    @property
    def functions(self):
        """ Function space of the decision process
        """
        return self._functions

    @property
    def state_variables(self):
        """ State space of the decision process
        """
        return self._state_variables

    @property
    def state_variable(self, id: str):
        """ State space of the decision process
        """
        for state_variable in self._state_variables:
            if state_variable.id == id:
                return state_variable
        raise KeyError(id)

    @property
    def action_variables(self):
        """ Action space of the decision process
        """
        return self._action_variables

    @property
    def action_variable(self, id: str):
        """ State space of the decision process
        """
        for action_variable in self._action_variables:
            if action_variable.id == id:
                return action_variable
        raise KeyError(id)

    @property
    def helper_variables(self):
        """ Helper space of the decision process
        """
        return self._helper_variables

    @property
    def helper_variable(self, id: str):
        """ State space of the decision process
        """
        for helper_variable in self._helper_variables:
            if helper_variable.id == id:
                return helper_variable
        raise KeyError(id)

    @property
    def parameters(self):
        """ Parameter space of the decision process
        """
        return self._parameters

    @property
    def parameter(self, id: str):
        """ State space of the decision process
        """
        for parameter in self._parameters:
            if parameter.id == id:
                return parameter
        raise KeyError(id)

    @property
    def dynamics_functions(self):
        """ Dynamics space of the decision process
        """
        return self._dynamics_functions

    @property
    def constraint_functions(self):
        """ Constraint space of the decision process
        """
        return self._constraint_functions

    @property
    def cost_functions(self) -> FrozenSet[CostFunction]:
        """ Cost function space of the decision process
        """
        return self._cost_functions

    def is_fully_known(self):
        """ Test whether all functions of the decision processes
            are known (i.e., described by symbolic expressions)
        """
        for f in self._functions:
            if not f.is_known():  # pragma: no cover
                return False  # pragma: no cover
        return True

    def validate(self) -> bool:
        """Validate the model

           The model will be validated if:
            - All variables/parameters are used in the model elsewhere,
            - All states variables are involved in the dynamics
              as state to be updated
            - All actions variables are involved in the dynamics
              updates


            Returns
            ---------
            bool
                True if the model is compatible with
                the abovementioned requirements

            Raises
            --------
            UselessVariableError
                If some variables are unused
            UselessParameterError
                If some parameters are unused
            DynamicsLessStateVariableError
                If any state variable is not updated
            NoCostFunctionError
                If no cost function has been defined
                when the decision process is not empty
        """

        # Empty decision processes are valid
        if (
            len(self._variables) == 0 and
            len(self._parameters) == 0 and
            len(self._functions) == 0
        ):
            return True

        if (len(self._cost_functions) == 0 and
           len(self._variables) > 0):
            error = "No cost function was defined. "
            error += "This is mandatory when variables are declared, "
            error += "even for stateless or constraint-free "
            error += "decision processes."
            raise NoCostFunctionError(error)

        all_vars_declared = self._variables
        all_params_declared = self._parameters
        all_vars_used = set()
        all_params_used = set()
        states_updates = set()
        state_variables_ids = {s.id for s in self._state_variables}
        all_vars_declared = {v.id for v in all_vars_declared}
        all_params_declared = {p.id for p in all_params_declared}
        for dyn in self._dynamics_functions:
            states_updates.add(dyn.state_var.container.id)
            d = dyn.state_var_update.get_data()
            variables = get_variables(d).union({dyn.state_var.container.id})
            all_vars_used = all_vars_used.union(variables)
            all_params_used = all_params_used.union(get_parameters(d))

        for cstr in self._constraint_functions:
            c = cstr.ineq.get_data()
            all_vars_used = all_vars_used.union(get_variables(c))
            all_params_used = all_params_used.union(get_parameters(c))

        for cost in self._cost_functions:
            r = cost.cost_expression.get_data()
            all_vars_used = all_vars_used.union(get_variables(r))
            all_params_used = all_params_used.union(get_parameters(r))

        useless_vars = all_vars_declared.difference(all_vars_used)
        if len(useless_vars) > 0:
            useless_vars = ",".join("'"+v+"'" for v in useless_vars)
            error = "{"+useless_vars+"}" +\
                    " variable(s) not used in the decision process." +\
                    " This is problematic since these variables are free" +\
                    " and thus can jeopardize the solution search process" +\
                    " of any end-controller"
            raise UselessVariableError(error)

        useless_params = all_params_declared.difference(all_params_used)
        if len(useless_params) > 0:
            self._parameters = {
                p for p in self._parameters if p not in useless_params
            }
            useless_params = ",".join("'"+p+"'" for p in useless_params)
            warning = "{"+useless_params+"}" +\
                      " parameters(s) not used in the decision process." +\
                      " It(they) will be removed from the decision process parameters."
            warnings.warn(warning)

        static_states = state_variables_ids.difference(states_updates)
        if len(static_states) > 0:
            static_states = ",".join("'"+s+"'" for s in static_states)
            error = "{"+static_states+"}" +\
                    " state variable(s) not used in any dynamics in the decision process." +\
                    " Please define one."
            raise DynamicsLessStateVariableError(error)

        self._validated = True
        self._create_data()
        return True

    def _create_data(self):
        state_vars_data = sorted([s.get_data()
                                 for s in self._state_variables],
                                 key=sorted_by_id)
        action_vars_data = sorted([u.get_data()
                                  for u in self._action_variables],
                                  key=sorted_by_id)
        helper_vars_data = sorted([h.get_data()
                                  for h in self._helper_variables],
                                  key=sorted_by_id)
        param_space_data = sorted([p.get_data()
                                  for p in self._parameters],
                                  key=sorted_by_id)
        dyn_functs_data = sorted([d.get_data()
                                  for d in self._dynamics_functions],
                                 key=sorted_by_id)
        cstr_functs_data = sorted([c.get_data()
                                   for c in self._constraint_functions],
                                  key=sorted_by_id)
        cstf_functs_data = sorted([r.get_data()
                                   for r in self._cost_functions],
                                  key=sorted_by_id)
        self._data = DecisionProcessData(id=self.id,
                                         description=self.description,
                                         state_variables=state_vars_data,
                                         action_variables=action_vars_data,
                                         helper_variables=helper_vars_data,
                                         parameters=param_space_data,
                                         dynamics_functions=dyn_functs_data,
                                         constraint_functions=cstr_functs_data,
                                         cost_functions=cstf_functs_data)

    def _add_state_variable(self, variable: Variable) -> Variable:
        """
            Add a variable to the state space

            Parameters
            ----------
            variable: Variable
                The variable

            Returns
            ----------
            Variable
                The same variable added
                to the state space

            Raises
            ----------
            AlreadyExistsError
                If the variable has already been added
                to the decision process

        """
        if variable in self._variables:
            raise AlreadyExistsError("State variable id "
                                     + variable.get_data().id
                                     + " is already used in"
                                     + " the whole variable space")
        self._state_variables.add(variable)
        self._variables.add(variable)
        return variable

    def add_state_variables(self,
                            *variables: Tuple[Variable, ...])\
            -> List[Variable]:
        """ Iterate over a list of variables
            to add them to the state space

            Parameters
            ----------
            variables: tuple of Variable
                Tuple of variables to add to the state space

            Returns
            ----------
            list of Variable
                The list of variables just added in the state space

            Raises
            ----------
            AlreadyExistsError
                If any state variable has already been added
                to the decision process


        """
        state_variables = []
        for v in variables:
            state_variables.append(self._add_state_variable(v))
        return state_variables

    def _add_action_variable(self, variable: Variable) -> Variable:
        """
            Add a variable to the action space

            Parameters
            ----------
            variable: Variable
                The variable

            Returns
            ----------
            Variable
                The same variable added
                to the action space

            Raises
            ----------
            AlreadyExistsError
                If the action variable has already been added
                to the decision process

        """
        if variable in self._variables:
            raise AlreadyExistsError("Action variable id "
                                     + variable.get_data().id
                                     + " is already used in"
                                     + " the whole variable space")
        self._action_variables.add(variable)
        self._variables.add(variable)
        return variable

    def add_action_variables(self,
                             *variables: Tuple[Variable, ...])\
            -> List[Variable]:
        """ Iterate over a list of variables
            to add them to the action space

            Parameters
            ----------
            variables: tuple of Variable
                Tuple of variables to add to the action space

            Returns
            ----------
            list of Variable
                The list of variables just added in the action space

            Raises
            ----------
            AlreadyExistsError
                If any action variable has already been added
                to the decision process


        """
        action_variables = []
        for v in variables:
            action_variables.append(self._add_action_variable(v))
        return action_variables

    def _add_helper_variable(self, variable: Variable) -> Variable:
        """
            Add a variable to the helper space

            Parameters
            ----------
            variable: Variable
                The variable

            Returns
            ----------
            Variable
                The same variable added
                to the helper space

            Raises
            ----------
            AlreadyExistsError
                If the helper variable has already been added
                to the decision process

        """
        if variable in self._variables:
            raise AlreadyExistsError("Helper variable id "
                                     + variable.get_data().id
                                     + " is already used in"
                                     + " the whole variable space")
        self._helper_variables.add(variable)
        self._variables.add(variable)
        return variable

    def add_helper_variables(self,
                             *variables: Tuple[Variable, ...])\
            -> List[Variable]:
        """ Iterate over a list of variables
            to add them to the helper space

            Parameters
            ----------
            variables: tuple of Variable
                Tuple of variables to add to the helper space

            Returns
            ----------
            list of Variable
                The list of variables just added in the helper space

            Raises
            ----------
            AlreadyExistsError
                If any helper variable has already been added
                to the decision process


        """
        helper_variables = []
        for v in variables:
            helper_variables.append(self._add_helper_variable(v))
        return helper_variables

    def _add_parameter(self, parameter: Parameter) -> Parameter:
        """
            Add a parameter to the parameter space

            Parameters
            ----------
            parameter: Parameter
                The parameter

            Returns
            ----------
            Parameter
                The same parameter added
                to the parameter space

            Raises
            ----------
            AlreadyExistsError
                If the parameter has already been added
                to the decision process

        """
        if parameter in self._parameters:
            raise AlreadyExistsError("Parameter "
                                     + parameter.get_data().id
                                     + " is already defined in"
                                     + " the parameter space")
        self._parameters.add(parameter)
        return parameter

    def add_parameters(self,
                       *parameters:
                       Tuple[Parameter, ...]) -> List[Parameter]:
        """ Iterate over a list of parameters
            to add them to the parameter space

            Parameters
            ----------
            parameters: tuple of Parameter
                Tuple of parameter to add to the parameter space

            Returns
            ----------
            list of Parameter
                The list of parameters just added in the parameter space

            Raises
            ----------
            AlreadyExistsError
                If any parameter has already been added
                to the decision process


        """
        params = []
        for p in parameters:
            params.append(self._add_parameter(p))
        return params

    def _add_dynamics_function(self, dynamics: Dynamics) -> Dynamics:
        """
            Add a dynamics to the dynamics space

            Parameters
            ----------
            dynamics: Dynamics
                The dynamics

            Returns
            ----------
            Variable
                The same dynamics added
                to the dynamics space

            Raises
            ----------
            AlreadyExistsError
                If the dynamics has already been added
                to the decision process

            NotDefinedVariable
                If any variable involved in the dynamics
                is not defined in the decision process

            NotDefinedParameter
                If any parameter involved in the dynamics
                is not defined in the decision process

            NotStateVariableError
                If the left-hand operand of the dynamics
                is not a state variable or is not defined
                at all

        """
        if dynamics in self._functions:
            raise AlreadyExistsError("Dynamics id"
                                     + dynamics.get_data().id
                                     + " is already defined in"
                                     + " the whole function space")

        left_var = dynamics.state_var
        self._verify_defined_variables(left_var + dynamics.state_var_update)
        self._verify_defined_parameters(dynamics.state_var_update)

        if left_var.container not in self._state_variables:
            error = left_var.container.id\
                    + " is not defined in the state variables."\
                    + " Use add_state_variable to add your "\
                    + " variable if this is intended."
            raise NotAStateVariableError(error)

        self._dynamics_functions.add(dynamics)
        self._functions.add(dynamics)
        return dynamics

    def add_dynamics_functions(self,
                               *dynamics:
                               Tuple[Dynamics, ...]) -> List[Dynamics]:
        """ Iterate over a list of dynamics
            to add them to the dynamics space

            Parameters
            ----------
            dynamics: tuple of Dynamics
                Tuple of dynamics to add to the dynamics space

            Returns
            ----------
            list of Dynamics
                The list of dynamics just added in the dynamics space

            Raises
            ----------
            AlreadyExistsError
                If any dynamics have already been added
                to the decision process

            NotDefinedVariable
                If any variable in any dynamics
                expression is not defined in the decision process

            NotDefinedParameter
                If any parameter in any dynamics
                is not defined in the decision process

            NotStateVariableError
                If the left-hand operand of any dynamics
                is not a state variable or is not defined
                at all

        """
        dynamics_functions = []
        for d in dynamics:
            dynamics_functions.append(self._add_dynamics_function(d))
        return dynamics_functions

    def _add_constraint_function(self, constraint: Constraint) -> Constraint:
        """
            Add a constraint to the constraint space

            Parameters
            ----------
            constraint: Constraint
                The constraint

            Returns
            ----------
            Constraint
                The same constraint added
                to the constraint space

            Raises
            ----------
            AlreadyExistsError
                If the constraint has already been added
                to the decision process

            NotDefinedVariable
                If any variable in the constraint
                expression is not defined in the decision process

            NotDefinedParameter
                If any parameter in the constraint
                expression is not defined in the decision process

        """
        if constraint in self._functions:
            raise AlreadyExistsError("Constraint id "
                                     + constraint.get_data().id
                                     + " is already defined in"
                                     + " the whole function space")

        self._verify_defined_parameters(constraint.ineq)
        self._verify_defined_variables(constraint.ineq)

        self._constraint_functions.add(constraint)
        self._functions.add(constraint)
        return constraint

    def add_constraint_functions(self,
                                 *constraints:
                                 Tuple[Constraint, ...]) -> List[Constraint]:
        """ Iterate over a list of constraint
            to add them to the constraint space

            Parameters
            ----------
            constraint: tuple of Constraint
                Tuple of constraint to add to the constraint space

            Returns
            ----------
            list of Constraint
                The list of constraints just added in the constraint space

            Raises
            ----------
            AlreadyExistsError
                If any of the constraints have already been added
                to the decision process

            NotDefinedVariable
                If any variable in any constraint
                expression is not defined in the decision process

            NotDefinedParameter
                If any parameter in any constraint
                expression is not defined in the decision process


        """
        constraint_functions = []
        for d in constraints:
            constraint_functions.append(self._add_constraint_function(d))
        return constraint_functions

    def _add_cost_function(self, cost_function: CostFunction) -> CostFunction:
        """
            Add a step cost function to the step cost space

            Parameters
            ----------
            costfunction: CostFunction
                The (instantaneous) cost function

            Returns
            ----------
            CostFunction
                The same step cost added
                to the step cost space

            Raises
            ----------
            AlreadyExistsError
                If the cost function has already been added
                to the decision process

            NotDefinedVariable
                If any variable in the cost expression
                is not defined in the decision process

            NotDefinedParameter
                If any parameter in the cost expression
                is not defined in the decision process

        """
        if cost_function in self._functions:
            raise AlreadyExistsError("Cost function id "
                                     + cost_function.get_data().id
                                     + " is already defined in"
                                     + " the whole function space")

        self._verify_defined_parameters(cost_function.cost_expression)
        self._verify_defined_variables(cost_function.cost_expression)

        self._cost_functions.add(cost_function)
        self._functions.add(cost_function)
        return cost_function

    def add_cost_functions(self,
                           *cost_functions:
                           Tuple[CostFunction, ...]) -> List[CostFunction]:
        """ Iterate over a list of cost functions
            to add them to the cost function space space

            Parameters
            ----------
            costfunctions: tuple of CostFunction
                Tuple of cost functions to add to the cost function space

            Returns
            ----------
            list of CostFunction
                The list of cost functions
                just added in the cost function space

            Raises
            ----------
            AlreadyExistsError
                If any of the cost functions have already been added
                to the decision process

            NotDefinedVariable
                If any variable in any of the cost expressions
                is not defined in the decision process

            NotDefinedParameter
                If any parameter in any of the cost expressions
                is not defined in the decision process

            NoVariableInvolvedError
                If any of the cost function do not involve any variable.


        """
        costfunctions_space = []
        for d in cost_functions:
            costfunctions_space.append(self._add_cost_function(d))
        return costfunctions_space

    def index_sets(self, attached=False) -> List[IndexSet]:
        """
            Extract all index sets involved in the decision process
        
            Returns
            ---------
            list of IndexSet
                List of index sets referenced through the decision process
        """
        if self._index_sets is not None and not attached:
            return self._index_sets
        from .utils.utils import free_index_sets, reduce_index_sets
        index_sets = []
        for v in self._variables:
            shape = v.get_data().shape
            if isinstance(shape, IndexSet) or (attached and len(shape) > 0):
                index_sets.append(shape)
            else:
                index_sets.extend(shape)
        for p in self._parameters:
            shape = p.get_data().shape
            if isinstance(shape, IndexSet) or (attached and len(shape) > 0):
                index_sets.append(shape)
            else:
                index_sets.extend(shape)

        for d in self._dynamics_functions:
            shape = d.state_var.get_data().container.shape
            if isinstance(shape, IndexSet) or (attached and len(shape) > 0):
                index_sets.append(shape)
            else:
                index_sets.extend(shape)
            index_sets.extend(free_index_sets(d.state_var_update.get_data(), attached))
            index_sets.extend(reduce_index_sets(d.state_var_update.get_data()))

        for c in self._constraint_functions:
            index_sets.extend(free_index_sets(c.ineq.get_data(), attached))
            index_sets.extend(reduce_index_sets(c.ineq.get_data()))

        for c in self._cost_functions:
            index_sets.extend(free_index_sets(c.cost_expression.get_data(), attached))
            index_sets.extend(reduce_index_sets(c.cost_expression.get_data()))

        if not attached:
            self._index_sets = list({(i.id): i for i in index_sets}.values())
        return list(index_sets)

    def get_data(self) -> DecisionProcessData:
        """
        Data Transferable Object (DTO)
        of DecisionProcess class

        Returns
        ---------
        DecisionProcessData
            DTO of the DecisionProcess object
        """
        if not self.validated:
            err = "DTO of the decision process object"
            err += " requested before validation."
            err += " Attempt to validate it before data creation"
            warnings.warn(err)
            self.validate()
        return self._data

    @classmethod
    def from_data(cls, data: DecisionProcessData) -> DecisionProcess:
        """
        Build an instance of the class `DecisionProcess`
        from a `DecisionProcessData` DTO.

        Parameters
        ---------
        :obj:`VariableData`
            DTO of the `Variable` object

        Returns
        ---------
        :obj:`Variable`
            Instance of `Variable` built with `data`


        """
        d = DecisionProcess(id=data.id, description=data.description)
        d.add_state_variables(*[Variable.from_data(v)
                              for v in data.state_variables])
        d.add_action_variables(*[Variable.from_data(v)
                               for v in data.action_variables])
        d.add_helper_variables(*[Variable.from_data(v)
                               for v in data.helper_variables])
        d.add_parameters(*[Parameter.from_data(p)
                         for p in data.parameters])
        d.add_dynamics_functions(*[Dynamics.from_data(d)
                                 for d in data.dynamics_functions])
        d.add_constraint_functions(*[Constraint.from_data(c)
                                   for c in data.constraint_functions])
        d.add_cost_functions(*[CostFunction.from_data(r)
                             for r in data.cost_functions])
        return d

    def _verify_defined_variables(self, expr: Expression) -> bool:
        """Check that every variables in 'expr' is defined
           in the variable space

           Attributes
           ----------
           expr : Expression
                An expression

           Raises
           ----------
           NotDefinedVariableError
                If any variable involved in 'expr'
                is not in the variable space

        """
        var_expr = get_variables(expr.get_data())
        all_vars_defined = {v.id for v in self._variables}
        left_vars = var_expr.difference(all_vars_defined)
        if len(left_vars) > 0:
            left_vars = ",".join("'"+v+"'" for v in left_vars)
            error = "{"+left_vars+"}" +\
                    " variable(s) not defined in the decision process." +\
                    " Use add_state_variable, add_action_variable" +\
                    " or even add_helper_variable if this is intended" +\
                    " (according to the role(s) of the variable(s))"
            raise NotDefinedVariableError(error)
        return True

    def _verify_defined_parameters(self, expr: Expression) -> bool:
        """Check that every parameters in 'expr' is defined
           in the parameter space

           Attributes
           ----------
           expr : Expression
                An expression

           Raises
           ----------
           NotDefinedParameterError
                If any parameter involved in 'expr'
                is not in the parameter space

        """
        param_expr = get_parameters(expr.get_data())
        param_space_ids = {p.id for p in self._parameters}
        left_params = param_expr.difference(param_space_ids)
        if len(left_params) > 0:
            left_params = ",".join("'"+p+"'" for p in left_params)
            error = "{"+left_params+"}" +\
                    " parameter(s) not defined in the decision process." +\
                    " Use add_parameter if this is intended"
            raise NotDefinedParameterError(error)
        return True
