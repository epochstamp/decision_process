from __future__ import annotations
from enum import Enum
from ...decision_process_components.evaluator import Evaluator
from ...decision_process_components.expression.numeric_expression\
    import NumericExpression, NumericExpressionData
from ...decision_process_components.expression.logical_expression\
    import LogicalExpression, LogicalExpressionData
from operator import ge, le, eq


class Ineqoperator(Enum):
    GE = ge  # Equality operator
    LE = le  # Less-or-equal operator
    EQ = eq  # Greater-or-equal operator

    def __repr__(self):
        if self.value == Ineqoperator.GE:
            return ">="
        elif self.value == Ineqoperator.LE:
            return "<="
        elif self.value == Ineqoperator.EQ:
            return "=="
        return str(self.value)


class IneqData(LogicalExpressionData):
    """
    DTO for LogicalExpression class.

    See LogicalExpression for more details
    about the attributes.
    """
    expr_1: NumericExpressionData
    ineq_op: Ineqoperator
    expr_2: NumericExpressionData


class Ineq(LogicalExpression):
    """ (In)equation between two expressions

        Attributes
        ----------
        expr_1: :obj:`Expression`
            First expression operand

        ineq_op: :obj:`InequalityOperator`
            Inequality operator

        expr_2: :obj:`Expression`
            Second expression operand
    """

    def __init__(self,
                 expr_1: NumericExpression,
                 ineq_op: Ineqoperator,
                 expr_2: NumericExpression):
        self._data = IneqData(expr_1=expr_1.get_data(),
                              ineq_op=ineq_op,
                              expr_2=expr_2.get_data())
        self._expr_1 = expr_1
        self._expr_2 = expr_2
        self._ineq_op = ineq_op

    def __call__(self, evaluator: Evaluator):
        """ Gives an evaluation out of the inequality operation
            using an object that implements `Evaluator`

            Parameters
            ----------
            evaluator: :obj:`Evaluator`

                An evaluator

            Returns
            ----------
            The result of the inequality operation evaluation

            Output type depends on the evaluator output.
        """
        eval_1 = self._expr_1(evaluator=evaluator)
        eval_2 = self._expr_2(evaluator=evaluator)
        return self._ineq_op.value(eval_1, eval_2)

    def get_data(self) -> IneqData:
        """
        Data Transferable Object (DTO) of `Ineq` class

        Returns
        ---------
        :obj:`IneqData`
            DTO of the `Ineq` object
        """
        return self._data

    @classmethod
    def from_data(cls, data: IneqData) -> Ineq:
        """
        Build an instance of the class Ineq
        from a IneqData DTO.

        Parameters
        ---------
        IneqData
            DTO of the Ineq object

        Returns
        ---------
        Ineq
            Instance of Ineq built with data


        """
        from ...utils.utils import get_expr_from_data
        return Ineq(expr_1=get_expr_from_data(data.expr_1),
                    ineq_op=data.ineq_op,
                    expr_2=get_expr_from_data(data.expr_2))

    def __repr__(self):
        return f"{self._expr_1.__repr__()} {self._ineq_op.__repr__()} {self._expr_2.__repr__()}"
