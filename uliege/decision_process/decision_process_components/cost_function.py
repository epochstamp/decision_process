# -*- coding: UTF-8 -*-
from __future__ import annotations
from typing import FrozenSet
from .function import Function
import numpy as np
from ..base.base_component import BaseComponent, BaseComponentData
from .expression.numeric_expression import\
    NumericExpression, NumericExpressionData
from ..utils.utils import free_index_sets, get_variables
from .variable import NoVariableInvolvedError
from .evaluator import Evaluator
from ..utils.utils import get_expr_from_data


class HorizonMask():
    """
        Base class for masking cost functions through a
        time horizon
    """

    def __call__(self,
                 current_time_step: float,
                 horizon_time: float) -> float:
        """
            Base callable method

            Parameters
            ----------
            current_time_step: float
                The current time step control. Always >0.
            current_time_step: float
                The time horizon. Might be infinite

            Returns
            ---------
            float
                The float coefficient applied to the cost function
        """
        raise NotImplementedError()


class DiscountFactor(HorizonMask):
    """
        Discount the cost function at time step t
        for an (in)finite horizon.

        (Useful for approximate solutions for infinite
         time horizon problems)

        Init function takes into parameter
        a discount factor to compute the following mask:

        m(t) = discount_factor**(t-1)
    """
    def __init__(self, discount_factor: float = 1.0):
        self._discount_factor = discount_factor

    def __call__(self,
                 current_time_step: float,
                 horizon_time: float) -> float:
        """
            Computes a discounted mask

            Parameters
            ----------
            current_time_step: float
                The current time step control
            current_time_step: float
                The time horizon. Might be infinite

            Returns
            ---------
            float
                discount_factor^(current_time_step-1)
        """
        return self._discount_factor**(current_time_step-1)


class LastTimeStepMask(HorizonMask):
    """
        Take into account only the last time step
        (e.g. for cost functions)

        (Useful for sparse-reward based decision processes)
    """

    def __call__(self,
                 current_time_step: float,
                 horizon_time: float) -> float:
        """
            Enable only the last time step before
            reaching horizon time control

            Parameters
            ----------
            current_time_step: float
                The current time step control
            current_time_step: float
                The time horizon. Might be infinite

            Returns
            ---------
            float
                1.0 if the current time step is the time horizon,
                0.0 otherwise
        """
        return 1.0 if current_time_step == horizon_time else 0.0


class UniformStepMask(HorizonMask):
    """
        Take into account all the step costs

        (Useful for dense-reward based decision processes)
    """

    def __call__(self,
                 current_time_step: float,
                 horizon_time: float) -> float:
        """
            Base callable method

            Parameters
            ----------
            current_time_step: float
                The current time step control
            current_time_step: float
                The time horizon. Might be infinite

            Returns
            ---------
            float
                The float coefficient applied to the cost functionAttributes
        ----------
        _discount_factor: The discount factor (private)
        """
        return 1.0


class CostFunctionData(BaseComponentData):
    """ DTO of CostFunction
        See CostFunction for more details about the attributes
    """
    class Config:
        arbitrary_types_allowed = True

    cost_expression: NumericExpressionData
    horizon_mask: HorizonMask


class FreeIndexError(BaseException):
    pass


class CostFunction(BaseComponent, Function):
    """ Cost function defined on
        the decision process variables and parameters.

        Attributes:
        -----------
        cost_expression : Expression
            The cost function expression

            /!\\ No index set is allowed in any expression
                 other than a reduce expression

        horizon_mask: HorizonMask (optional, default to UniformStepMask)
            A function that returns a coefficient
            to apply to the cost function with
            the current and the horizon time step

            Note: The second argument of the mask function
            is always greater or equal to the first one

        Raises
        ----------
        FreeIndexError
            If cost_expression contains any free index
            (i.e., an index set which lies outside a reduce operator)

        NoVariableInvolvedError
            If no variable is used inside the cost function
    """
    def __init__(self,
                 cost_expression: NumericExpression,
                 id: str = "",
                 description: str = "",
                 horizon_mask: HorizonMask = UniformStepMask()):
        BaseComponent.__init__(self, id=id, description=description)
        cost_expr_data = cost_expression.get_data()
        free_idxs = free_index_sets(cost_expr_data, attached=False)
        self._variables_ids = get_variables(cost_expression.get_data())
        if len(self._variables_ids) == 0:
            error = "No variable is involved in the current cost function."\
                    + "This is problematic in a decision process since"\
                    + "any solution will be optimal."
            raise NoVariableInvolvedError(error)

        if len(free_idxs) > 0:
            error = "Cost functions cannot have free indexes.\n"
            error += "Here are the spotted free indexes: \n"
            lst_free_idxs = [str(i[0].id) for i in free_idxs]
            error += "".join(lst_free_idxs)

            raise FreeIndexError(error)

        self._data = CostFunctionData(id=id,
                                      description=description,
                                      cost_expression=cost_expr_data,
                                      horizon_mask=horizon_mask)
        self._cost_expression = cost_expression
        self._horizon_mask = horizon_mask
        

    @property
    def variables_ids(self) -> FrozenSet[str]:
        return self._variables_ids

    @property
    def cost_expression(self) -> NumericExpression:
        return self._cost_expression

    @property
    def horizon_mask(self) -> HorizonMask:
        return self._horizon_mask

    def is_known(self) -> bool:
        return True

    def __call__(self,
                 evaluator: Evaluator,
                 t: float = 0,
                 T: float = np.inf):
        """
        Execute the step cost function at time t
        in a time horizon T
        """
        if t > T:
            temp = t
            t = T
            T = temp
        return (self.horizon_mask(t, T) *
                self.cost_expression(evaluator))

    def get_data(self) -> CostFunctionData:
        """
        Data Transferable Object (DTO) of CostFunction class

        Returns
        ---------
        CostFunctionData
            DTO of the CostFunction object
        """
        return self._data

    @classmethod
    def from_data(cls, data: CostFunctionData) -> CostFunction:
        """
        Build an instance of the class `CostFunction`
        from a `CostFunctionData` DTO.

        Parameters
        ---------
        :obj:`VariableData`
            DTO of the `Variable` object

        Returns
        ---------
        :obj:`Variable`
            Instance of `Variable` built with `data`


        """
        expr_from_data = get_expr_from_data(data.cost_expression)
        return CostFunction(id=data.id,
                            cost_expression=expr_from_data,
                            description=data.description,
                            horizon_mask=data.horizon_mask)
