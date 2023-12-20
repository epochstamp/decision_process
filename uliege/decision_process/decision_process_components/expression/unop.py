from __future__ import annotations
from enum import Enum
from ..evaluator import Evaluator
from ..expression.numeric_expression\
    import NumericExpression, NumericExpressionData
from operator import neg


class Unoperator(Enum):
    NEG = neg  # Negation
    ABS = abs  # Absolute value


unoperator_str = {
    Unoperator.NEG: "-",
    Unoperator.ABS: "abs"
}


class UnopData(NumericExpressionData):
    """ `Unop` DTO

         See `Unop` for more details on attributes
    """
    expr: NumericExpressionData
    unop: Unoperator


class Unop(NumericExpression):

    """ Unary operator applied to an expression
        Attributes
        ----------
        expr: :obj:`NumericExpression`
            Expression operand

        unop: :obj:`Binoperator`
            Unary operator
    """

    def __init__(self,
                 expr: NumericExpression,
                 unop: Unoperator) -> None:
        self._data = UnopData(expr=expr.get_data(), unop=unop)
        self._expr = expr
        self._unop = unop

    @property
    def expr(self):
        return self._expr

    def __call__(self, evaluator: Evaluator):
        """ Gives an evaluation out of the unary operation
            applied to an expression
            using an object that implements `Evaluator`

            Parameters
            ----------
            evaluator: :obj:`Evaluator`

                An evaluator

            Returns
            ----------
            The result of the expression evaluation

            Output type depends on the evaluator output.
        """
        return self._unop.value(self._expr(evaluator=evaluator))

    def get_data(self) -> UnopData:
        """
        Data Transferable Object (DTO) of `Unop` class

        Returns
        ---------
        :obj:`UnopData`
            DTO of the `Unop` object
        """
        return self._data

    @classmethod
    def from_data(cls, data: UnopData) -> Unop:
        """
        Build an instance of the class `Unop`
        from a `UnopData` DTO.

        Parameters
        ---------
        :obj:`UnopData`
            DTO of the `Unop` object

        Returns
        ---------
        :obj:`Unop`
            Instance of `Unop` built with `data`


        """
        from ...utils.utils import get_expr_from_data
        return Unop(expr=get_expr_from_data(data.expr),
                    unop=data.unop)

    def __repr__(self):
        return f"{unoperator_str[self._unop]}({self._expr.__repr__()})"
