from uliege.decision_process.decision_process_components.expression.indexed_container import\
                                                              IndexedContainer
from uliege.decision_process.decision_process_components.variable import Variable
from uliege.decision_process.decision_process_components.expression.index_set import\
                                                 IndexSet
from uliege.decision_process.decision_process_components.expression.binop import Binoperator,\
                                                         BinopData
from uliege.decision_process.decision_process_components.expression.unop import Unoperator,\
                                                        UnopData
from uliege.decision_process.decision_process_components.expression.reduce import Reduce,\
                                                          Reducer,\
                                                          ReduceIndexSet,\
                                                          ReduceData,\
                                                          sum_reduce,\
                                                          prod_reduce
from uliege.decision_process.decision_process_components.expression.numeric_expression import\
    NumericExpression
from uliege.decision_process.decision_process_components.expression.ineq import Ineqoperator,\
                                                        IneqData, Ineq
from hypothesis import assume, given
from hypothesis.strategies import builds,\
                                  sampled_from,\
                                  one_of,\
                                  lists,\
                                  recursive
import random
import numpy as np
from typing import Tuple
from .utils import reasonable_lenghty_text,\
                        build_unoperator_strat,\
                        build_binoperator_strat,\
                        build_reduce_strat,\
                        build_indexed_parameter,\
                        build_indexed_variable
from uliege.decision_process.utils.utils import get_expr_from_data

np.random.seed(1000)
random.seed(1000)


@given(reasonable_lenghty_text,
       reasonable_lenghty_text,
       sampled_from(Binoperator))
def test_binoperation_variables(id_1: str,
                                id_2: str,
                                b_op: Binoperator):
    """
    Test binary operation between simple variables

    Parameters
    ----------
    id_1, id_2: str
        Ids of the (two) Variable object(s)
        (see Variable docstring)

    b_op: Binoperator
        Binary operator

    Asserts
    ---------
        Successfully created a binary operation between simple variables

    """
    v1 = Variable(id=id_1)
    v2 = Variable(id=id_2)
    binoperation = b_op.value(v1, v2)
    ic1 = IndexedContainer(container=v1).get_data()
    ic2 = IndexedContainer(container=v2).get_data()
    binoperation_test = BinopData(expr_1=ic1, binop=b_op, expr_2=ic2)
    assert(binoperation.get_data() == binoperation_test)


@given(reasonable_lenghty_text,
       reasonable_lenghty_text,
       sampled_from(Binoperator),
       lists(builds(IndexSet, id=reasonable_lenghty_text),
             min_size=1,
             max_size=2,
             unique_by=lambda x: x.id),
       lists(builds(IndexSet, id=reasonable_lenghty_text),
             min_size=1,
             max_size=2,
             unique_by=lambda x: x.id))
def test_binoperation_indexed_variables(id_1: str,
                                        id_2: str,
                                        b_op: Binoperator,
                                        shape_1: Tuple[IndexSet, ...],
                                        shape_2: Tuple[IndexSet, ...]):
    """
    Test binary operation between indexed variables

    Parameters
    ----------
    id_1, id_2: str
        Ids of the (two) Variable object(s)
        (see Variable docstring)

    shape_1, shape_2: tuple of IndexSet
        Shapes of the two variables
        (sets may be differents)

    b_op: Binoperator
        Binary operator

    Assumes
    ---------
        - Shapes are of the same length

    Asserts
    ---------
        Successfully created a binary operation between indexed variables

    """
    assume(len(shape_1) == len(shape_2))
    shape_1 = tuple(shape_1)
    shape_2 = tuple(shape_2)
    v1 = Variable(id=id_1, shape=shape_1)
    v2 = Variable(id=id_2, shape=shape_2)
    binoperation = b_op.value(v1[shape_1], v2[shape_2])
    ic1 = IndexedContainer(container=v1,
                           idx_seq=shape_1).get_data()
    ic2 = IndexedContainer(container=v2,
                           idx_seq=shape_2).get_data()
    binoperation_test = BinopData(expr_1=ic1, binop=b_op, expr_2=ic2)
    assert(binoperation.get_data() == binoperation_test)


@given(reasonable_lenghty_text,
       sampled_from(Unoperator))
def test_unoperation_variables(id: str,
                               u_op: Unoperator):
    """
    Test unary operation on a simple variable

    Parameters
    ----------
    id: str
        Ids of the (two) Variable object(s)
        (see Variable docstring)

    u_op: Unoperator
        Unary operator

    Asserts
    ---------
        Successfully created an unary operation over a simple variable

    """
    v = Variable(id=id)
    unoperation = u_op.value(v)
    ic = IndexedContainer(container=v).get_data()
    unoperation_test = UnopData(expr=ic, unop=u_op)
    assert(unoperation.get_data() == unoperation_test)


@given(reasonable_lenghty_text,
       sampled_from(Unoperator),
       lists(builds(IndexSet, id=reasonable_lenghty_text),
             min_size=1,
             max_size=2,
             unique_by=lambda x: x.id))
def test_unoperation_indexed_variables(id: str,
                                       u_op: Unoperator,
                                       shape: Tuple[IndexSet, ...]):
    """
    Test unary operation on a simple variable

    Parameters
    ----------
    id: str
        Ids of the (two) Variable object(s)
        (see Variable docstring)

    u_op: Unoperator
        Unary operator

    shape: tuple of IndexSet
        Shapes of the variable
        (sets may be differents)

    Asserts
    ---------
        Successfully created an unary operation over a simple variable

    """
    shape = tuple(shape)
    v = Variable(id=id, shape=shape)
    unoperation = u_op.value(v[shape])
    ic = IndexedContainer(container=v,
                          idx_seq=shape).get_data()
    unoperation_test = UnopData(expr=ic, unop=u_op)
    assert(unoperation.get_data() == unoperation_test)


@given(reasonable_lenghty_text,
       builds(IndexSet,
              id=reasonable_lenghty_text))
def test_sum_indexset_variables(id: str,
                                idx_set: IndexSet):
    """
    Test sum reduce over a simple variable

    (This is equivalent to multiply the variable by
     the cardinality of the index set)

    Parameters
    ----------
    id: str
        Id of the Variable object
        (see Variable docstring)

    idx_set: IndexSet
        An index set

    Asserts
    ---------
        Successfully created an sum reduce
        operation over a simple variable

    """
    v = Variable(id=id, shape=idx_set)
    ic = IndexedContainer(container=v, idx_seq=idx_set)
    sum_expr = sum_reduce(inner_expr=ic, idx_set=idx_set)
    red_idx_set = ReduceIndexSet(reducer=Reducer.SUM,
                                 idx_set=idx_set)
    reduce_test = Reduce(inner_expr=ic,
                         idx_reduce_set=red_idx_set).get_data()
    assert(sum_expr.get_data() == reduce_test)


@given(reasonable_lenghty_text,
       builds(IndexSet,
              id=reasonable_lenghty_text))
def test_prod_indexset_variables(id: str,
                                 idx_set: IndexSet):
    """
    Test sum reduce over a simple variable

    (This is equivalent to multiply the variable by
     the cardinality of the index set)

    Parameters
    ----------
    id: str
        Id of the Variable object
        (see Variable docstring)

    idx_set: IndexSet
        An index set

    Asserts
    ---------
        Successfully created an sum reduce
        operation over a simple variable

    """
    v = Variable(id=id, shape=idx_set)
    ic = IndexedContainer(container=v, idx_seq=idx_set)
    prod_expr = prod_reduce(inner_expr=ic, idx_set=idx_set)
    red_idx_set = ReduceIndexSet(reducer=Reducer.PROD,
                                 idx_set=idx_set)
    reduce_test = Reduce(inner_expr=ic,
                         idx_reduce_set=red_idx_set).get_data()
    assert(prod_expr.get_data() == reduce_test)


@given(reasonable_lenghty_text,
       reasonable_lenghty_text,
       sampled_from(Ineqoperator))
def test_ineqoperation_variables(id_1: str,
                                 id_2: str,
                                 i_op: Ineqoperator):
    """
    Test binary operation between simple variables

    Parameters
    ----------
    id_1, id_2: str
        Ids of the (two) Variable object(s)
        (see Variable docstring)

    i_op: Ineqoperator
        Inequality operator

    Asserts
    ---------
        Successfully created a inequality operation between simple variables

    """
    v1 = Variable(id=id_1)
    v2 = Variable(id=id_2)
    ineqoperation = i_op.value(v1, v2)
    ic1 = IndexedContainer(container=v1).get_data()
    ic2 = IndexedContainer(container=v2).get_data()
    ineqoperation_test = IneqData(expr_1=ic1, ineq_op=i_op, expr_2=ic2)
    assert(ineqoperation.get_data() == ineqoperation_test)


@given(reasonable_lenghty_text,
       reasonable_lenghty_text,
       sampled_from(Ineqoperator),
       lists(builds(IndexSet, id=reasonable_lenghty_text),
             min_size=1,
             max_size=2,
             unique_by=lambda x: x.id),
       lists(builds(IndexSet, id=reasonable_lenghty_text),
             min_size=1,
             max_size=2,
             unique_by=lambda x: x.id))
def test_ineqoperation_indexed_variables(id_1: str,
                                         id_2: str,
                                         i_op: Ineqoperator,
                                         shape_1: Tuple[IndexSet, ...],
                                         shape_2: Tuple[IndexSet, ...]):
    """
    Test binary operation between indexed variables

    Parameters
    ----------
    id_1, id_2: str
        Ids of the (two) Variable object(s)
        (see Variable docstring)

    shape_1, shape_2: tuple of IndexSet
        Shapes of the two variables
        (sets may be differents)

    i_op: Ineqoperator
        Inequality operator

    Assumes
    ---------
        - Shapes are of the same length

    Asserts
    ---------
        Successfully created an inequality operation
        between indexed variables

    """
    assume(len(shape_1) == len(shape_2))
    shape_1 = tuple(shape_1)
    shape_2 = tuple(shape_2)
    v1 = Variable(id=id_1, shape=shape_1)
    v2 = Variable(id=id_2, shape=shape_2)
    ineqoperation = i_op.value(v1[shape_1], v2[shape_2])
    ic1 = IndexedContainer(container=v1,
                           idx_seq=shape_1).get_data()
    ic2 = IndexedContainer(container=v2,
                           idx_seq=shape_2).get_data()
    ineqoperation_test = IneqData(expr_1=ic1, ineq_op=i_op, expr_2=ic2)
    assert(ineqoperation.get_data() == ineqoperation_test)


@given(recursive(base=one_of(build_indexed_parameter(),
                             build_indexed_variable()),
                 extend=lambda s: one_of(build_binoperator_strat(s)(),
                                         build_unoperator_strat(s)(),
                                         build_reduce_strat(s)()),
                 max_leaves=8),
       recursive(base=one_of(build_indexed_parameter(),
                             build_indexed_variable()),
                 extend=lambda s: one_of(build_binoperator_strat(s)(),
                                         build_unoperator_strat(s)(),
                                         build_reduce_strat(s)()),
                 max_leaves=8),
       sampled_from(Binoperator))
def test_complex_binoperation(expr1: NumericExpression,
                              expr2: NumericExpression,
                              binoperator: Binoperator):
    binop_expr = binoperator.value(expr1, expr2)
    binop_data = BinopData(expr_1=expr1.get_data(),
                           binop=binoperator,
                           expr_2=expr2.get_data())
    assert(binop_expr.get_data() == binop_data)


@given(recursive(base=one_of(build_indexed_parameter(),
                             build_indexed_variable()),
                 extend=lambda s: one_of(build_binoperator_strat(s)(),
                                         build_unoperator_strat(s)(),
                                         build_reduce_strat(s)()),
                 max_leaves=8),
       sampled_from(Unoperator))
def test_complex_unoperation(expr: NumericExpression,
                             unoperator: Unoperator):
    unop_expr = unoperator.value(expr)
    unop_data = UnopData(expr=expr.get_data(),
                         unop=unoperator)
    assert(unop_expr.get_data() == unop_data)


@given(recursive(base=one_of(build_indexed_parameter(),
                             build_indexed_variable()),
                 extend=lambda s: one_of(build_binoperator_strat(s)(),
                                         build_unoperator_strat(s)(),
                                         build_reduce_strat(s)()),
                 max_leaves=8),
       builds(IndexSet, id=reasonable_lenghty_text))
def test_complex_sum_reduce(inner_expr: NumericExpression,
                            idx_set: IndexSet):
    sum_expr = sum_reduce(inner_expr, idx_set)
    idx_reduce_set = ReduceIndexSet(reducer=Reducer.SUM,
                                    idx_set=idx_set)
    sum_data = ReduceData(inner_expr=inner_expr.get_data(),
                          idx_reduce_set=idx_reduce_set)
    assert(sum_expr.get_data() == sum_data)


@given(recursive(base=one_of(build_indexed_parameter(),
                             build_indexed_variable()),
                 extend=lambda s: one_of(build_binoperator_strat(s)(),
                                         build_unoperator_strat(s)(),
                                         build_reduce_strat(s)()),
                 max_leaves=8),
       builds(IndexSet, id=reasonable_lenghty_text))
def test_complex_prod_reduce(inner_expr: NumericExpression,
                             idx_set: IndexSet):
    sum_expr = prod_reduce(inner_expr, idx_set)
    idx_reduce_set = ReduceIndexSet(reducer=Reducer.PROD,
                                    idx_set=idx_set)
    sum_data = ReduceData(inner_expr=inner_expr.get_data(),
                          idx_reduce_set=idx_reduce_set)
    assert(sum_expr.get_data() == sum_data)


@given(recursive(base=one_of(build_indexed_parameter(),
                             build_indexed_variable()),
                 extend=lambda s: one_of(build_binoperator_strat(s)(),
                                         build_unoperator_strat(s)(),
                                         build_reduce_strat(s)()),
                 max_leaves=8),
       recursive(base=one_of(build_indexed_parameter(),
                             build_indexed_variable()),
                 extend=lambda s: one_of(build_binoperator_strat(s)(),
                                         build_unoperator_strat(s)(),
                                         build_reduce_strat(s)()),
                 max_leaves=8),
       sampled_from(Ineqoperator))
def test_complex_inequation(expr1: NumericExpression,
                            expr2: NumericExpression,
                            ineqoperator: Ineqoperator):
    ineq_expr = ineqoperator.value(expr1, expr2)
    ineq_data = IneqData(expr_1=expr1.get_data(),
                         ineq_op=ineqoperator,
                         expr_2=expr2.get_data())
    assert(ineq_expr.get_data() == ineq_data)


@given(recursive(base=one_of(build_indexed_parameter(),
                             build_indexed_variable()),
                 extend=lambda s: one_of(build_binoperator_strat(s)(),
                                         build_unoperator_strat(s)(),
                                         build_reduce_strat(s)()),
                 max_leaves=8))
def test_numerical_expression_from_data(expr: NumericExpression):
    expr_data = expr.get_data()
    expr_from_data = get_expr_from_data(expr.get_data())
    assert(expr_data == expr_from_data.get_data())


@given(recursive(base=one_of(build_indexed_parameter(),
                             build_indexed_variable()),
                 extend=lambda s: one_of(build_binoperator_strat(s)(),
                                         build_unoperator_strat(s)(),
                                         build_reduce_strat(s)()),
                 max_leaves=8),
       recursive(base=one_of(build_indexed_parameter(),
                             build_indexed_variable()),
                 extend=lambda s: one_of(build_binoperator_strat(s)(),
                                         build_unoperator_strat(s)(),
                                         build_reduce_strat(s)()),
                 max_leaves=8),
       sampled_from(Ineqoperator))
def test_inequation_from_data(expr1: NumericExpression,
                              expr2: NumericExpression,
                              ineqoperator: Ineqoperator):
    ineq_expr = ineqoperator.value(expr1, expr2)
    ineq_data = IneqData(expr_1=expr1.get_data(),
                         ineq_op=ineqoperator,
                         expr_2=expr2.get_data())
    ineq_from_data = Ineq.from_data(ineq_expr.get_data())
    assert(ineq_from_data.get_data() == ineq_data)
