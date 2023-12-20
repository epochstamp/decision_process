from __future__ import annotations
from pydantic import BaseModel
from enum import Enum


from ...decision_process_components.evaluator import Evaluator, IndexSequenceDoesNotExists, IndexSetNotFoundError, ValueNotFoundError
from ...decision_process_components.expression.numeric_expression import\
    NumericExpression, NumericExpressionData
from ...decision_process_components.variable import Variable
from ...decision_process_components.parameter import Parameter
from operator import add, mul
from ...decision_process_components.expression.index_set import IndexSet,\
                                                             Index
from typing import Tuple, List, Union


class Reducer(Enum):
    SUM = add  # Sum reducer
    PROD = mul  # Prod reducer


class ReduceWentWrong(BaseException):
    pass


class ReduceIndexSet(BaseModel):
    """Index reducer (e.g., sum, product...)
       Is directly a Data Transferable Object (DTO)

       Parameters
       ----------

       reducer: :obj:`Reducer`
            Type of the reducer

       idx_set: :obj:`IndexSet`
            The index set to iterate over
    """
    reducer: Reducer = Reducer.SUM
    idx_set: IndexSet


class ReduceData(NumericExpressionData):
    """ `Reduce` DTO
         See `Reduce` for more details on attributes
    """
    inner_expr: NumericExpressionData
    idx_reduce_set: ReduceIndexSet


class ReduceOverriderEvaluator(Evaluator):

    """ Overrides an `Evaluator` object.

        Attributes
        ----------
        inner_evaluator: :obj:`En`
            Inner expression of the reducer
        surcharge_idx: :obj:`tuple`
            Reduce operation on the inner expression
    """

    def __init__(self,
                 inner_evaluator: Evaluator,
                 surcharge_idx: Tuple[IndexSet, Index]):
        self._evaluator = inner_evaluator
        if isinstance(self._evaluator, ReduceOverriderEvaluator):
            self._previous_indexes = self._evaluator._previous_indexes + [self._evaluator._surcharge_idx]
        else:
            self._previous_indexes = []
        self._surcharge_idx = surcharge_idx

    def _get_index(self, index: IndexSet) -> List[Index]:
        """Overrides the index set when it corresponds
           to the index of a reduce operator.

        """
        if index == self._surcharge_idx[0]:
            return self._surcharge_idx[1]
        else:
            return self._evaluator.get_index(index)

    def _get(self,
             container: Union[Variable, Parameter],
             idx_seq: List[Index]):
        """See `Evaluator` docstring

        """
        return self._evaluator.get(container, idx_seq)

    def _get_all_components_by_index(self,
                                     index_set: IndexSet,
                                     indexes_source: List[Index]) -> List[Index]:
        """See `Evaluator` docstring

        """
        return self._evaluator.get_all_components_by_index(index_set, indexes_source)


class Reduce(NumericExpression):

    """ Reduce an inner expression using an operator and an index set

        Attributes
        ----------
        inner_expr: :obj:`Expression`
            Inner expression of the reducer
        idx_reduce_set: :obj:`ReduceIndexSet`
            Reduce operation on the inner expression
    """

    def __init__(self,
                 inner_expr: NumericExpression,
                 idx_reduce_set: ReduceIndexSet) -> None:
        self._data = ReduceData(inner_expr=inner_expr.get_data(),
                                idx_reduce_set=idx_reduce_set)
        self._inner_expr = inner_expr
        from ...utils.utils import free_index_sets
        self._inner_index_sets = list(free_index_sets(
            inner_expr.get_data(), attached=True, keep_partial_free_if_attached=True
        ))
        self._inner_index_sets.sort(key=len, reverse=True)

    def __call__(self, evaluator: Evaluator):
        """ Gives an evaluation out of reduce operation
            over the inner expression
            using an object that implements `Evaluator`.

            Parameters
            ----------
            evaluator: :obj:`Evaluator`

                An evaluator

            Returns
            ----------
            The result of the expression evaluation

            Output type depends on the evaluator output.
        """
        acc = None
        op = self._data.idx_reduce_set.reducer
        index_set = self._data.idx_reduce_set.idx_set
        inner_index_sets_without_reduce_idx_set = []
        for idx_seq in self._inner_index_sets:
            if len(idx_seq) > 1:
                try:
                    inner_index_sets_without_reduce_idx_set = [
                        (idx_set, evaluator.get_index(idx_set)) for idx_set in idx_seq if idx_set != index_set
                    ]
                    break
                except IndexSetNotFoundError:
                    pass
        index_components = evaluator.get_all_components_by_index(
            index_set, inner_index_sets_without_reduce_idx_set
        )
        for index in index_components:
            # Surcharge the evaluator to replace
            # index_iter by an index component
            evaluator = ReduceOverriderEvaluator(inner_evaluator=evaluator,
                                                 surcharge_idx=(index_set, index))
            value = self._inner_expr(evaluator=evaluator)
            if acc is None:
                acc = value
            else:
                acc = op.value(acc, value)
        if acc is None:
            acc = 0
        return acc

    def get_data(self) -> ReduceData:
        """
        Data Transferable Object (DTO) of `Reduce` class

        Returns
        ---------
        :obj:`ReduceData`
            DTO of the `Reduce` object
        """
        return self._data

    @classmethod
    def from_data(cls, data: ReduceData) -> Reduce:
        """
        Build an instance of the class `Reduce`
        from a `ReduceData` DTO.

        Parameters
        ---------
        :obj:`ReduceData`
            DTO of the `Reduce` object

        Returns
        ---------
        :obj:`Reduce`
            Instance of `Reduce` built with `data`


        """
        from ...utils.utils import get_expr_from_data
        return Reduce(inner_expr=get_expr_from_data(data.inner_expr),
                      idx_reduce_set=data.idx_reduce_set)


def sum_reduce(inner_expr: NumericExpression,
               idx_set: IndexSet) -> Reduce:
    """
        Return a sum reducer operator over an index set

        Parameters
        ----------
        inner_expr: Expression
            The inner expression

        idx_set: IndexSet
            The index set to iterate over

        Returns
        ---------
        Reduce
            A sum reducer operator
    """
    return Reduce(inner_expr=inner_expr,
                  idx_reduce_set=ReduceIndexSet(reducer=Reducer.SUM,
                                                idx_set=idx_set))


def prod_reduce(inner_expr: NumericExpression,
                idx_set: IndexSet) -> Reduce:
    """
        Return a prod reducer operator over an index set

        Parameters
        ----------
        inner_expr: Expression
            The inner expression

        idx_set: IndexSet
            The index set to iterate over

        Returns
        ---------
        Reduce
            A prod reducer operator
    """
    return Reduce(inner_expr=inner_expr,
                  idx_reduce_set=ReduceIndexSet(reducer=Reducer.PROD,
                                                idx_set=idx_set))

def __repr__(self):
    if self._data.idx_reduce_set.reducer == Reducer.MUL:
        operator = "mul"
    else:
        operator = "sum"
    return f"{operator}_[{self._data.idx_reduce_set.idx_set.id}]" + "{" + self._inner_expr.__repr__() + "}"
