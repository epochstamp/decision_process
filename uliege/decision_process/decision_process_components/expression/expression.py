from pydantic import BaseModel
from ...decision_process_components.expression.operand import Operand
from ...base.transferable import Transferable
from abc import abstractmethod


class ExpressionData(BaseModel):
    pass


class Expression(Operand, Transferable):
    """Decision process expression interface
    """
    from ...decision_process_components.evaluator import Evaluator

    @abstractmethod
    def __call__(self, evaluator: Evaluator):
        """ Gives an evaluation out of the expression
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
        raise NotImplementedError()
