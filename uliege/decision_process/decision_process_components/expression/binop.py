from __future__ import annotations
from enum import Enum
from ...decision_process_components.expression.numeric_expression\
    import NumericExpression, NumericExpressionData
from ...decision_process_components.evaluator import Evaluator
from operator import mul, add, truediv, sub


class Binoperator(Enum):
    MUL = mul  # Product operator
    DIV = truediv  # Division operator
    SUB = sub  # Difference operator
    ADD = add  # Addition operator


binoperator_str = {
    Binoperator.MUL: "*",
    Binoperator.ADD: "+",
    Binoperator.SUB: "-",
    Binoperator.DIV: "/"
}


class BinopData(NumericExpressionData):
    """ `Binop` DTO
         See `Binop` for more details on attributes
    """
    expr_1: NumericExpressionData
    binop: Binoperator
    expr_2: NumericExpressionData


class Binop(NumericExpression):

    """ Binary operation between two expressions

        Attributes
        ----------
        expr_1: :obj:`Expression`
            First expression operand

        binop: :obj:`Binoperator`
            Binary operator

        expr_2: :obj:`Expression`
            Second expression operand
    """

    def __init__(self,
                 expr_1: NumericExpression,
                 binop: Binoperator,
                 expr_2: NumericExpression) -> None:
        self._data = BinopData(expr_1=expr_1.get_data(),
                               binop=binop,
                               expr_2=expr_2.get_data())
        self._expr_1 = expr_1
        self._expr_2 = expr_2
        self._binop = binop

    @property
    def expr_2(self):
        return self._expr_2

    def __call__(self, evaluator: Evaluator):
        """ Gives an evaluation out of the binary operation
            between two operations
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
        eval_1 = self._expr_1(evaluator=evaluator)
        if (isinstance(eval_1, int) or isinstance(eval_1, float)) and eval_1 == 0 and (self._binop == Binoperator.MUL
                            or self._binop == Binoperator.DIV):
            return 0
        eval_2 = self._expr_2(evaluator=evaluator)
        return self._binop.value(eval_1, eval_2)

    def get_data(self) -> BinopData:
        """
        Data Transferable Object (DTO) of `Binop` class

        Returns
        ---------
        :obj:`BinopData`
            DTO of the `Binop` object
        """
        return self._data

    @classmethod
    def from_data(cls, data: BinopData) -> Binop:
        """
        Build an instance of the class Binop
        from a BinopData DTO.

        Parameters
        ---------
        BinopData
            DTO of the Binop object

        Returns
        ---------
        Binop
            Instance of Binop built with data


        """
        from ...utils.utils import get_expr_from_data
        return Binop(expr_1=get_expr_from_data(data.expr_1),
                     binop=data.binop,
                     expr_2=get_expr_from_data(data.expr_2))


    def __repr__(self):
        return f"{self._expr_1.__repr__()} {binoperator_str[self._binop]} {self.expr_2.__repr__()}"
