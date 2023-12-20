from ...decision_process_components.expression.expression import Expression,\
                                                              ExpressionData


class NumericExpressionData(ExpressionData):
    """
    DTO for NumericExpression class.
    Only for specialization purpose.
    """
    pass


class NumericExpression(Expression):
    """
    Base class for numeric expressions.
    (e.g. variables, parameters, add, sub...)
    Only for specialization purpose.
    """
    pass
