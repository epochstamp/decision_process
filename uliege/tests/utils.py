from __future__ import annotations
from hypothesis.strategies import composite,\
                                  sampled_from,\
                                  builds,\
                                  text,\
                                  lists,\
                                  one_of,\
                                  floats,\
                                  integers,\
                                  recursive,\
                                  none,\
                                  dictionaries,\
                                  just
from hypothesis import assume
from uliege.decision_process import (
    Binop,
    Binoperator,
    Unop,
    Unoperator,
    Reduce,
    Reducer,
    ReduceIndexSet,
    sum_reduce,
    Index,
    IndexSet,
    Parameter,
    ParameterData,
    Variable,
    VariableData,
    Type,
    is_real,
    is_integer,
    IndexedContainer,
    Evaluator,
    DecisionProcess,
    Datastream,
    DataError,
    Dynamics,
    Constraint,
    CostFunction,
    Ineqoperator,
)
from typing import Union, List, Tuple, Dict, FrozenSet, Callable
from uliege.decision_process.utils import utils
import numpy as np
import itertools

reasonable_lenghty_text = text(list('abcdefghijklmnopqrstuvwxyz_'),
                               min_size=2,
                               max_size=5)


def dict_with_maps(draw):
    d = draw(dictionaries(reasonable_lenghty_text,
                          lists(reasonable_lenghty_text,
                                min_size=0,
                                max_size=10),
                          min_size=1,
                          max_size=10))
    return list(d.items()), d


dict_with_maps = composite(dict_with_maps)


def nested_dict_with_maps(dictstrat):
    def nested_dict_with_list_of_keys(draw):
        maps, nested_dict = draw(dictstrat)
        d = dict()
        k = draw(reasonable_lenghty_text)
        assume(k not in [m[0] for m in maps])
        d[k] = nested_dict
        return maps + [(k, nested_dict)], d
    return composite(nested_dict_with_list_of_keys)


def give_birth_to_index_set(idxstrat):
    def thats_a_beautiful_index_set_maam(draw) -> List[IndexSet]:
        index_sets = draw(idxstrat)
        if not isinstance(index_sets, list):
            index_sets = [index_sets]
        if np.any([isinstance(i, list) for i in index_sets]):
            index_sets = np.array(index_sets).flat
        lst = []
        for parent in index_sets:
            if isinstance(parent, list):
                parent = np.random.choice(parent)
            lst.append(IndexSet(id=draw(reasonable_lenghty_text),
                                parent=parent))
        return lst
    return composite(thats_a_beautiful_index_set_maam)


def dict_product(dicts):
    """
        Cartesian product of dicts

        Parameters
        -----------
        dicts: list of dict
            Dictionaries

        Returns
        -----------
        list of dicts
    """
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def flatten(items, seqtypes=(list, tuple)):
    for i, _ in enumerate(items):
        while i < len(items) and isinstance(items[i], seqtypes):
            items[i:i+1] = items[i]
    return items


def merge_dict(dict1, dict2):
    for key, val in dict1.items():
        if type(val) == dict:
            if key in dict2 and type(dict2[key] == dict):
                merge_dict(dict1[key], dict2[key])
        else:
            if key in dict2:
                dict1[key] = dict2[key]

    for key, val in dict2.items():
        if key not in dict1:
            dict1[key] = val

    return dict1


def three_random_partition(lst):
    x = np.asarray(lst)
    np.random.shuffle(x)
    length = x.shape[0]
    weights = np.sort(np.random.rand(2)+0.1)
    weights /= sum(weights)
    return (x[: int(length * weights[0])],
            x[int(length * weights[0]): int(length * weights[1])],
            x[int(weights[1]):])


def build_scalar_variable(draw) -> IndexedContainer:
    v_type = draw(sampled_from(Type))
    return IndexedContainer(container=Variable(v_type=v_type))


build_scalar_variable = composite(build_scalar_variable)


def build_scalar_parameter(draw) -> IndexedContainer:
    return IndexedContainer(Parameter())


build_scalar_parameter = composite(build_scalar_parameter)


def build_unoperator_strat(opestrat):
    def build_unoperation(draw) -> Unop:
        operator = draw(sampled_from(Unoperator))
        expr = draw(opestrat)
        return Unop(expr=expr,
                    unop=operator)
    return composite(build_unoperation)


def build_binoperator_strat(opestrat):
    def build_binoperation(draw) -> Binop:
        operator = draw(sampled_from(Binoperator))
        expr_1 = draw(opestrat)
        expr_2 = draw(opestrat)
        return Binop(expr_1=expr_1,
                     binop=operator,
                     expr_2=expr_2)
    return composite(build_binoperation)


def build_reduce_strat(opestrat):
    def build_reduce(draw) -> Reduce:
        operator = draw(sampled_from(Reducer))
        inner_expr = draw(opestrat)
        idx_set = draw(builds(IndexSet,
                              id=reasonable_lenghty_text))
        r_idx_set = ReduceIndexSet(reducer=operator,
                                   idx_set=idx_set)
        return Reduce(inner_expr=inner_expr,
                      idx_reduce_set=r_idx_set)
    return composite(build_reduce)


def build_indexed_parameter(draw) -> IndexedContainer:
    idx_seq = tuple(draw(lists(builds(IndexSet,
                                      id=reasonable_lenghty_text),
                               min_size=1,
                               max_size=2,
                               unique_by=lambda x: x.id)))
    return IndexedContainer(Parameter(shape=idx_seq),
                            idx_seq=idx_seq)


build_indexed_parameter = composite(build_indexed_parameter)


def build_indexed_variable(draw) -> IndexedContainer:
    v_type = draw(sampled_from(Type))
    idx_seq = tuple(draw(lists(builds(IndexSet,
                                      id=reasonable_lenghty_text),
                               min_size=1,
                               max_size=2,
                               unique_by=lambda x: x.id)))
    return IndexedContainer(Variable(v_type=v_type,
                                     shape=idx_seq),
                            idx_seq=idx_seq)


build_indexed_variable = composite(build_indexed_variable)


def build_indexed_real_variable(draw) -> IndexedContainer:
    idx_seq = tuple(draw(lists(builds(IndexSet,
                                      id=reasonable_lenghty_text),
                               min_size=1,
                               max_size=2,
                               unique_by=lambda x: x.id)))
    return IndexedContainer(Variable(shape=idx_seq,
                                     support=(-100000, 100000)),
                            idx_seq=idx_seq)


build_indexed_real_variable = composite(build_indexed_real_variable)


class SimpleEvaluator(Evaluator):

    def __init__(self,
                 draw,
                 idx_container: IndexedContainer,
                 map_type_to_value: Dict[Type, float],
                 value_parameter: float):
        self.eval_dict = dict()
        if type(idx_container.container) is Variable:
            type_container = "var"
            v_type = idx_container.container.get_data().v_type
            value = map_type_to_value[v_type]
        else:
            type_container = "param"
            value = value_parameter
        self.eval_dict[type_container] = dict()
        self.index_set_map = dict()
        self.eval_dict[type_container][idx_container.container] = dict()
        idx_seq_container = idx_container.get_data().idx_seq
        self.index_set_val = dict()
        for idx_set in idx_seq_container:
            lst_idx_names = draw(
                            lists(reasonable_lenghty_text,
                                  min_size=1,
                                  max_size=5)
                            )
            lst_idxs = [Index(id=i) for i in lst_idx_names]
            self.index_set_map[idx_set] = lst_idxs
            valuation = np.random.choice(lst_idxs)
            self.index_set_val[idx_set] = valuation
        eval_dict = self.eval_dict[type_container][idx_container.container]
        for idx_map in dict_product(self.index_set_map):
            idx_seq = list()
            for idx_set in idx_seq_container:
                idx_seq.append(idx_map[idx_set])
            eval_dict[tuple(idx_seq)] = value

    def _get(self,
             container: Union[Variable, Parameter],
             idx_seq: List[Index]) -> float:
        type_container = "var" if type(container) is Variable else "param"
        return self.eval_dict[type_container][container][tuple(idx_seq)]

    def _get_index(self, index_set: IndexSet) -> Index:
        return self.index_set_val[index_set]

    def _get_all_components_by_index(self,
                                     index_set: IndexSet) -> List[Index]:
        return self.index_set_map[index_set]

    def get_index_sets(self):
        return set(self.index_set_map.keys())


def build_idx_container_evaluator(draw) -> Tuple[IndexedContainer,
                                                 Evaluator,
                                                 Dict[Type, float],
                                                 float, float]:
    idx_container = draw(one_of(build_indexed_variable(),
                                build_indexed_parameter()))

    map_type_to_value = dict()
    for t in Type:

        low = max(t.value[0], -1000)
        high = min(t.value[1], 1000)
        random_value = draw(floats(min_value=low,
                                   max_value=high,
                                   allow_nan=False,
                                   allow_infinity=False))
        if random_value == 0:
            random_value = 1.0
        if (t == Type.INTEGER
            or t == Type.NON_NEGATIVE_INTEGER
                or t == Type.NON_POSITIVE_INTEGER):
            random_value = np.round(random_value)
        map_type_to_value[t] = random_value
    value_parameter = draw(floats(min_value=low,
                                  max_value=high,
                                  allow_nan=False,
                                  allow_infinity=False))
    if value_parameter == 0:
        value_parameter = 1.0
    return (idx_container,
            SimpleEvaluator(draw,
                            idx_container,
                            map_type_to_value,
                            value_parameter),
            map_type_to_value,
            value_parameter)


build_idx_container_evaluator = composite(build_idx_container_evaluator)


class MergedEvaluator(SimpleEvaluator):

    def __init__(self,
                 evaluator_1: SimpleEvaluator,
                 evaluator_2: Evaluator,
                 map_type_value: Dict[Type, float],
                 value_parameter: float
                 ):
        self.index_set_map = merge_dict(evaluator_1.index_set_map,
                                        evaluator_2.index_set_map)
        self.index_set_val = merge_dict(evaluator_1.index_set_val,
                                        evaluator_2.index_set_val)
        self.eval_dict = merge_dict(evaluator_1.eval_dict,
                                    evaluator_2.eval_dict)

        # Loop through variables
        for v in self.eval_dict.get("var", {}).keys():
            index_set_map_v = {idx_set: self.index_set_map[idx_set]
                               for idx_set in v.shape}
            for idx_map in dict_product(index_set_map_v):
                idx_seq = list()
                for idx_set in v.shape:
                    idx_seq.append(idx_map[idx_set])
                if tuple(idx_seq) not in self.eval_dict["var"][v]:
                    type_value = map_type_value[v.get_data().v_type]
                    self.eval_dict["var"][v][tuple(idx_seq)] = type_value

        # Loop through parameters
        for p in self.eval_dict.get("param", {}).keys():
            index_set_map_p = {idx_set: self.index_set_map[idx_set]
                               for idx_set in p.shape}
            for idx_map in dict_product(index_set_map_p):
                idx_seq = list()
                for idx_set in p.shape:
                    idx_seq.append(idx_map[idx_set])
                if tuple(idx_seq) not in self.eval_dict["param"]:
                    v_param = value_parameter
                    self.eval_dict["param"][p][tuple(idx_seq)] = v_param


def combine_evaluators(evaluator_1: SimpleEvaluator,
                       evaluator_2: SimpleEvaluator,
                       map_type: Dict[Type, float],
                       value_parameter: float) -> SimpleEvaluator:
    return MergedEvaluator(evaluator_1,
                           evaluator_2,
                           map_type,
                           value_parameter)


def build_unoperator_eval_strat(opestrat):
    def build_unoperation(draw) -> Tuple[Unop,
                                         Evaluator,
                                         Dict[Type, float],
                                         float]:
        operator = draw(sampled_from(Unoperator))
        expr, evaluator, map_type, param_value = draw(opestrat)
        return (Unop(expr=expr,
                     unop=operator),
                evaluator,
                map_type,
                param_value)
    return composite(build_unoperation)


def build_binoperator_eval_strat(opestrat):
    def build_binoperation(draw) -> Tuple[Binop,
                                          Evaluator,
                                          Dict[Type, float],
                                          float]:
        operator = draw(sampled_from(Binoperator))
        expr_1, eval_1, map_type, param_value = draw(opestrat)
        expr_2, eval_2, _, _ = draw(opestrat)
        combined_evaluator = combine_evaluators(eval_1,
                                                eval_2,
                                                map_type,
                                                param_value)
        if operator == Binoperator.DIV:
            assume(abs(expr_2(eval_2)) > 0)
        return (Binop(expr_1=expr_1,
                      binop=operator,
                      expr_2=expr_2),
                combined_evaluator,
                map_type,
                param_value)
    return composite(build_binoperation)


def build_reduce_eval_strat(opestrat):
    def build_reduce(draw) -> Tuple[Reduce,
                                    Evaluator,
                                    Dict[Type, float],
                                    float]:
        operator = draw(sampled_from(Reducer))
        inner_expr, eval, map_type, param_value = draw(opestrat)
        idx_set = np.random.choice(list(eval.get_index_sets()))
        r_idx_set = ReduceIndexSet(reducer=operator,
                                   idx_set=idx_set)
        return (Reduce(inner_expr=inner_expr,
                       idx_reduce_set=r_idx_set),
                eval,
                map_type,
                param_value)
    return composite(build_reduce)


def build_operator_strat(opestrat):
    def build_operator(draw):
        return draw(one_of(build_unoperator_strat(opestrat)(),
                           build_binoperator_strat(opestrat)()))
    return composite(build_operator)


def build_decision_process(draw) -> DecisionProcess:

    state_variables = draw(one_of(none(),
                                  lists(build_scalar_variable(),
                                        min_size=1,
                                        max_size=2)))
    if state_variables is None:
        state_variables = []
    action_variables = draw(one_of(none(),
                                   lists(build_scalar_variable(),
                                         min_size=1,
                                         max_size=2)))
    if action_variables is None:
        action_variables = []
    variables = state_variables + action_variables
    helper_variables = []
    if len(variables) == 0:
        helper_variables = draw(lists(build_scalar_variable(),
                                      min_size=1,
                                      max_size=2))
        variables += helper_variables
    parameters = draw(one_of(none(),
                             lists(build_scalar_parameter(),
                                   min_size=1,
                                   max_size=2)))
    if parameters is None:
        parameters = []
    l_vars = variables
    l_params = parameters

    dynamics_functions = set()
    for s in state_variables:
        if len(l_params) == 0:
            base_strat = sampled_from(l_vars)
        else:
            base_strat = one_of(sampled_from(l_vars),
                                sampled_from(l_params))
        dyn_expr = draw(recursive(base=base_strat,
                                  extend=lambda s:
                                  build_operator_strat(s)(),
                                  max_leaves=2))
        dynamics_functions.add(Dynamics(s, dyn_expr))
    used_variables = set()
    used_parameters = set()
    constraint_functions = set()
    cost_functions = set()
    while (l_vars != [] or
           l_params != []
           or len(cost_functions) == 0):
        l_vars_sample = l_vars if l_vars != [] else variables
        l_params_sample = l_params if l_params != [] else parameters
        if len(l_params_sample) > 0:
            base_strat = one_of(sampled_from(l_vars_sample),
                                sampled_from(l_params_sample))
        else:
            base_strat = sampled_from(l_vars_sample)
        constr_expr_1 = draw(recursive(base=sampled_from(l_vars_sample),
                                       extend=lambda s:
                                       build_operator_strat(s)(),
                                       max_leaves=2))
        constr_expr_2 = draw(recursive(base=base_strat,
                                       extend=lambda s:
                                       build_operator_strat(s)(),
                                       max_leaves=2))
        ineq_op = draw(sampled_from(Ineqoperator))
        full_ineq = ineq_op.value(constr_expr_1, constr_expr_2)
        constraint_functions.add(Constraint(full_ineq))

        cost_expr = draw(recursive(base=sampled_from(l_vars_sample),
                                   extend=lambda s:
                                   build_operator_strat(s)(),
                                   max_leaves=4))
        cost_functions.add(CostFunction(cost_expr))
        used_variables = {*used_variables,
                          *utils.get_variables(cost_expr.get_data()),
                          *utils.get_variables(full_ineq.get_data())}
        used_parameters = {*used_parameters,
                           *utils.get_parameters(cost_expr.get_data()),
                           *utils.get_parameters(full_ineq.get_data())}
        l_vars = [v for v in l_vars
                  if v.container.id not in used_variables]
        l_params = [p for p in l_params
                    if p.container.id not in used_parameters]
    decision_process = DecisionProcess()
    decision_process.add_state_variables(*[s.container
                                           for s in state_variables])
    decision_process.add_action_variables(*[a.container
                                            for a in action_variables])
    decision_process.add_helper_variables(*[h.container
                                            for h in helper_variables])
    decision_process.add_parameters(*[p.container
                                      for p in parameters])
    decision_process.add_dynamics_functions(*dynamics_functions)
    decision_process.add_constraint_functions(*constraint_functions)
    decision_process.add_cost_functions(*cost_functions)
    return decision_process


build_decision_process = composite(build_decision_process)


def build_unoperator_strat_detailed(opestrat):
    def build_unoperation(draw) -> Tuple[Unop,
                                         List[IndexedContainer],
                                         List[IndexedContainer],
                                         FrozenSet[IndexSet]]:
        operator = draw(sampled_from(Unoperator))
        expr, lst_vars, lst_params, index_sets = draw(opestrat)
        unop_expr = Unop(expr=expr,
                         unop=operator)
        return unop_expr, lst_vars, lst_params, index_sets
    return composite(build_unoperation)


def build_binoperator_strat_detailed(opestrat):
    def build_binoperation(draw) -> Tuple[Binop,
                                          List[IndexedContainer],
                                          List[IndexedContainer],
                                          FrozenSet[IndexSet]]:
        operator = draw(sampled_from(Binoperator))
        (expr_1, lst_vars_1, lst_params_1,
         (free_idx_sets_1, reduce_idx_sets_1)) = draw(opestrat)
        (expr_2, lst_vars_2, lst_params_2,
         (free_idx_sets_2, reduce_idx_sets_2)) = draw(opestrat)
        binop_expr = Binop(expr_1=expr_1,
                           binop=operator,
                           expr_2=expr_2)
        return (binop_expr,
                lst_vars_1 + lst_vars_2,
                lst_params_1 + lst_params_2,
                (free_idx_sets_1.union(free_idx_sets_2),
                 reduce_idx_sets_1.union(reduce_idx_sets_2)))
    return composite(build_binoperation)


def build_reduce_strat_detailed(opestrat):
    def build_reduce(draw) -> Tuple[Reduce,
                                    List[IndexedContainer],
                                    List[IndexedContainer],
                                    FrozenSet[IndexSet]]:
        operator = draw(sampled_from(Reducer))
        (inner_expr, inner_vars, inner_params,
         (free_idx_sets, reduce_idx_sets)) = draw(opestrat)
        idx_set = draw(builds(IndexSet,
                              id=reasonable_lenghty_text))
        r_idx_set = ReduceIndexSet(reducer=operator,
                                   idx_set=idx_set)
        reduce_expr = Reduce(inner_expr=inner_expr,
                             idx_reduce_set=r_idx_set)
        return (reduce_expr, inner_vars, inner_params,
                (free_idx_sets.difference({idx_set}),
                 reduce_idx_sets.union({idx_set})))
    return composite(build_reduce)


def build_indexed_parameter_detailed(draw) -> Tuple[IndexedContainer,
                                                    List[IndexedContainer],
                                                    List[IndexedContainer],
                                                    FrozenSet[IndexSet]]:
    idx_seq = tuple(draw(lists(builds(IndexSet,
                                      id=reasonable_lenghty_text),
                               min_size=1,
                               max_size=3,
                               unique_by=lambda x: x.id)))
    idx_container = IndexedContainer(Parameter(shape=idx_seq),
                                     idx_seq=idx_seq)
    return (idx_container,
            [],
            [idx_container],
            ({*idx_seq}, set()))


build_indexed_parameter_detailed = composite(build_indexed_parameter_detailed)


def build_indexed_variable_detailed(draw) -> Tuple[IndexedContainer,
                                                   List[IndexedContainer],
                                                   List[IndexedContainer],
                                                   FrozenSet[IndexSet]]:
    v_type = draw(sampled_from(Type))
    idx_seq = tuple(draw(lists(builds(IndexSet,
                                      id=reasonable_lenghty_text),
                               min_size=1,
                               max_size=3,
                               unique_by=lambda x: x.id)))
    idx_container = IndexedContainer(Variable(v_type=v_type,
                                              shape=idx_seq),
                                     idx_seq=idx_seq)
    return (idx_container,
            [idx_container],
            [],
            ({*idx_seq}, set()))


build_indexed_variable_detailed = composite(build_indexed_variable_detailed)


class TestDatastream(Datastream):

    def __init__(self,
                 state_vars: List[Variable],
                 params: List[Parameter],
                 index_sets: List[IndexSet],
                 hypothesis_drawer: Callable):
        self._index_set_components = dict()
        draw = hypothesis_drawer
        for index_set in index_sets:
            self._index_set_components[index_set] =\
                draw(lists(builds(Index, id=reasonable_lenghty_text),
                           min_size=1,
                           max_size=5))

        self._init_vars = dict()
        kwargs_allow = {"allow_infinity": False, "allow_nan": False}
        for v in state_vars:
            kwargs_bounds = {"min_value": max(-100000, v.support[0]),
                             "max_value": min(100000, v.support[1])}
            kwargs_all = {**kwargs_bounds, **kwargs_allow}
            shape = v.shape
            number_strat = floats if is_real(v.v_type) else integers
            kwargs = kwargs_all if number_strat == floats\
                else kwargs_bounds
            if shape == tuple():
                self._init_vars[v.id] = draw(number_strat(**kwargs))
                continue
            index_set_comps = {i: self.get_indexes_by_index_set(i)
                               for i in shape}
            index_set_mappings = (dict(zip(index_set_comps, x))
                                  for x in itertools.product(
                                      *index_set_comps.values()))
            for idx_set_mapping in index_set_mappings:
                indexes = tuple([idx_set_mapping[i] for i in shape])
                self._init_vars[(v.id,) + indexes] =\
                    draw(number_strat(**kwargs))

        self._maximum_time_horizon = draw(integers(min_value=1, max_value=100))
        self._params = dict()
        for p in params:
            shape = p.shape
            if shape == tuple():
                self._params[p.id] = draw(number_strat(**kwargs)) + 1.0
                continue
            index_set_comps = {i: self.get_indexes_by_index_set(i)
                               for i in shape}
            index_set_mappings = (dict(zip(index_set_comps, x))
                                  for x in itertools.product(
                                      *index_set_comps.values()))
            for idx_set_mapping in index_set_mappings:
                indexes = tuple([idx_set_mapping[i] for i in shape])
                self._params[(p.id,) + indexes] =\
                    draw(floats(
                         min_value=1,
                         max_value=100000,
                         allow_infinity=False,
                         allow_nan=False))

    @property
    def maximum_time_horizon(self):
        return self._maximum_time_horizon

    def _get_initialization(self,
                            var_data: VariableData,
                            idx_seq: List[Index]) -> Union[int,
                                                           float,
                                                           None]:
        key = var_data.id if var_data.shape == tuple() else\
              (var_data.id,) + tuple(idx_seq)
        value = self._init_vars.get(key, None)
        if value is None:
            raise DataError(str(key) + " variable init not found")
        return value

    def _get_parameter(self,
                       param_data: ParameterData,
                       idx_seq: List[Index],
                       length: int) -> Union[List[int],
                                             List[float],
                                             float,
                                             int,
                                             None]:
        key = param_data.id if param_data.shape == tuple() else\
              (param_data.id,) + tuple(idx_seq)
        p_vec = self._params.get(key, None)
        if p_vec is None:
            raise DataError(str(key) + " parameter value(s) not found")
        elif isinstance(self._params[key], list):
            p_vec = self._params.get(key, None)
            return p_vec if p_vec is None\
                else p_vec[:min(length, self.maximum_time_horizon)]
        else:
            return [self._params[key]]*min(length, self.maximum_time_horizon)

    def _get_indexes_by_index_set(self,
                                  index_set: IndexSet) -> Union[None,
                                                                List[Index]]:

        indexes = self._index_set_components.get(index_set, None)
        if indexes is None:
            raise DataError(
                "Indexes of the index set "+index_set.id+" not found"
            )
        return indexes


def build_unconstrained_decision_process_with_datastream(draw) ->\
        Tuple[DecisionProcess, Datastream]:

    state_variables = draw(lists(build_indexed_real_variable(),
                                 min_size=1,
                                 max_size=6))
    state_variables = [s.container for s in state_variables]
    if state_variables is None:
        state_variables = []
    action_variables =\
        [Variable(shape=draw(one_of(just(v.shape), just(tuple()))))
            for v in state_variables]
    parameters = [Parameter(shape=draw(one_of(just(v.shape), just(tuple()))))
                  for v in state_variables]

    dynamics_functions = set()
    cost_functions = set()
    for i in range(len(state_variables)):
        s = state_variables[i]
        a = action_variables[i]
        p = parameters[i]
        shape = s.shape if s.shape != tuple() else\
            (a.shape if a.shape != tuple() else
             p.shape)
        s_indexed = s[shape] if s.shape != tuple() else s
        a_indexed = a[shape] if a.shape != tuple() else a
        p_indexed = p[shape] if p.shape != tuple() else p
        expr_update = s_indexed + a_indexed*p_indexed
        dynamics_functions.add(Dynamics(s_indexed, expr_update))
        pcost = Parameter(shape=draw(one_of(just(s.shape), just(tuple()))))
        parameters.append(pcost)
        pcost_indexed = pcost[shape] if pcost.shape != tuple() else pcost
        cost_expr = pcost_indexed * s_indexed
        for indexset in utils.free_index_sets(cost_expr.get_data()):
            cost_expr = sum_reduce(inner_expr=cost_expr, idx_set=indexset)
        cost_functions.add(CostFunction(cost_expression=cost_expr))
    decision_process = DecisionProcess()
    decision_process.add_state_variables(*state_variables)
    decision_process.add_action_variables(*action_variables)
    decision_process.add_parameters(*parameters)
    decision_process.add_dynamics_functions(*dynamics_functions)
    decision_process.add_cost_functions(*cost_functions)
    shapes = [v.shape for v in state_variables]
    dp_data_stream = TestDatastream(state_vars=state_variables,
                                    params=parameters,
                                    index_sets=[i for shape in shapes
                                                for i in shape],
                                    hypothesis_drawer=draw)
    return decision_process, dp_data_stream


build_unconstrained_decision_process_with_datastream =\
    composite(build_unconstrained_decision_process_with_datastream)
