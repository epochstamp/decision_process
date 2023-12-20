from ...decision_process_components.expression.expression import Expression,\
                                                              ExpressionData
from ...base.transferable import Transferable


class LogicalExpressionData(ExpressionData):
    """
    DTO for LogicalExpression class.
    Only for specialization purpose.
    """
    pass


class LogicalExpression(Expression, Transferable):
    """
    Base class for numeric expressions.
    (e.g. <=, >=, ==, ifs...)
    Only for specialization purpose.
    """
    pass
