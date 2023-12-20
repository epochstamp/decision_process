from __future__ import annotations

from ..utils.utils import get_variables
from .function import Function
from .evaluator import Evaluator
from .variable import Variable, VariableData
from .expression.numeric_expression\
    import NumericExpression, NumericExpressionData
from ..base.base_component import BaseComponent, BaseComponentData
from .expression.indexed_container import\
                                                        IndexedContainer,\
                                                        IndexedContainerData,\
                                                        IndexedVariableData
from typing import FrozenSet, Union


class DynamicsData(BaseComponentData):
    """ DTO of `Dynamics`
        See `Dynamics` docstring for more details

    """

    state_var: IndexedVariableData
    state_var_update: NumericExpressionData


class Dynamics(Function, BaseComponent):

    """ Defines an executable and model-based dynamics

        Attributes
        ----------
        id: :obj:`str`
            The id of the dynamics

        state_var: :obj:`IndexedContainer`
            A indexed variable

        state_var_update: :obj:`Expression`
            A numeric expression which express
            the state update (or transition)

        Raises
        ---------
        ValueError
            If the variable is not shape-free or not fully indexed
            (i.e., the indexation should make it a scalar value)
        TypeError
            If state_var is not an indexed variable


    """

    def __init__(self,
                 state_var: Union[Variable, IndexedContainer],
                 state_var_update: NumericExpression,
                 id: str = "",
                 description: str = ""):
        super().__init__(id=id)
        if isinstance(state_var, Variable):
            shape = state_var.shape
            if not isinstance(shape, tuple) or len(shape) > 0:
                error = "Variable must be shape-free or fully indexed"
                raise ValueError(error)
            else:
                state_var = IndexedContainer(container=state_var)
        elif isinstance(state_var, IndexedContainer):
            container = state_var.container
            if not isinstance(container, Variable):
                raise TypeError("state_var must be an (indexed) variable")
        state_var_update_data = state_var_update.get_data()
        self._data = DynamicsData(id=id,
                                  description=description,
                                  state_var=state_var.get_data(),
                                  state_var_update=state_var_update_data)
        self._state_var = state_var
        self._state_var_update = state_var_update
        self._variables_ids = get_variables(self._state_var_update.get_data())

    @property
    def variables_ids(self) -> FrozenSet[str]:
        return self._variables_ids

    def __call__(self, evaluator: Evaluator):
        return self.state_var_update(evaluator)

    @property
    def state_var(self) -> IndexedContainer:
        """ State variable of the dynamics
        """
        return self._state_var

    @property
    def state_var_update(self) -> NumericExpression:
        """ State variable update
        """
        return self._state_var_update

    def is_known(self) -> bool:
        """`Constraint` formula is known

        Returns
        -------
        True
        """
        return True

    def get_data(self) -> DynamicsData:
        """
        Data Transferable Object (DTO) of the class
        that implements this interface

        Returns
        ---------
        :obj:`DynamicsData`
            DTO of the object
        """
        return self._data

    @classmethod
    def from_data(cls, data: DynamicsData) -> Dynamics:
        """
        Build an instance of the class that implements
        this interface from the proper DTO

        Parameters
        ---------
        :obj:`Constraint`
            DTO of the object

        Returns
        ---------
        :obj:


        """
        state_var = IndexedContainer.from_data(data.state_var)

        from ..utils.utils import get_expr_from_data
        state_var_update = get_expr_from_data(data.state_var_update)
        return Dynamics(id=data.id,
                        description=data.description,
                        state_var=state_var,
                        state_var_update=state_var_update)
