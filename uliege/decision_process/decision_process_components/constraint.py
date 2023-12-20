from __future__ import annotations
from typing import FrozenSet

from pydantic.errors import FrozenSetError
from .function import Function
from .evaluator import Evaluator
from ..base.base_component import BaseComponent, BaseComponentData
from .expression.ineq import Ineq,\
                                                        IneqData
from ..utils.utils import flatten, free_index_sets, get_variables, index_sets
from .variable import NoVariableInvolvedError


class ConstraintData(BaseComponentData):
    """ DTO of `Constraint` class
        See `Constraint` docstring for more details

    """

    ineq: IneqData


class IndexSetMismatchBothSidesConstraint(BaseException):
    pass


class Constraint(Function, BaseComponent):

    """ Defines an executable and model-based constraint

        Attributes
        ----------
        id: :obj:`str`
            The id of the constraint

        ineq: :obj:`Ineq`
            Inequation expression

        shift_state: :obj:`bool`
            Whether the states involved in the constraints are considered at time step t or t-1 (with t > 0)

        Raises
        ----------
        NoVariableInvolvedError
            If no variable is used in the constraint expression

    """

    def __init__(self,
                 ineq: Ineq,
                 id: str = "",
                 description: str = "",
                 shift_time_state: bool = False):
        super().__init__(id=id, description=description)
        self._variables_ids = get_variables(ineq.get_data())
        if len(self._variables_ids) == 0:
            error = "No variable is involved in the current constraint. "\
                    + "This is problematic in a decision process since "\
                    + "the parameters are fixed in advance."
            raise NoVariableInvolvedError(error)
        self._free_idx_sets = list(free_index_sets(ineq.get_data(), attached=True))
        self._free_idx_sets.sort(key=len, reverse=True)
        self._idx_sets_left = list(free_index_sets(ineq.get_data().expr_1, attached=True))
        self._idx_sets_right = list(free_index_sets(ineq.get_data().expr_2, attached=True))
        idx_sets_left_unattached = free_index_sets(ineq.get_data().expr_1, attached=False)
        idx_sets_right_unattached = free_index_sets(ineq.get_data().expr_2, attached=False)
        if (
            not idx_sets_left_unattached.issubset(idx_sets_right_unattached) and
            not idx_sets_right_unattached.issubset(idx_sets_left_unattached)
        ):
            error = "Set of index sets of one side of the constraint should be equal (or included) in the index sets of the other side "
            error += "(either left to right or right to left) but it is not the case of this constraint. "
            error += f"Index set sequence of both sides are {idx_sets_left_unattached} and {idx_sets_right_unattached}, respectively."
            raise IndexSetMismatchBothSidesConstraint(error)
        self._data = ConstraintData(id=id,
                                    description=description,
                                    ineq=ineq.get_data())
        self._ineq = ineq
        self._shift_time_state = shift_time_state

    @property
    def shift_time_state(self) -> bool:
        return self._shift_time_state

    @property
    def variables_ids(self) -> FrozenSet[str]:
        return self._variables_ids

    @property
    def ineq(self) -> Ineq:
        """ Constraint inequality
        """
        return self._ineq

    @property
    def free_idx_sets(self):
        return self._free_idx_sets

    @property
    def idx_sets_left(self):
        return self._idx_sets_left

    @property
    def idx_sets_right(self):
        return self._idx_sets_right

    def __call__(self, evaluator: Evaluator):
        return self.ineq(evaluator)

    def is_known(self) -> bool:
        """`Constraint` formula is known

        Returns
        -------
        True
        """
        return True

    def get_data(self) -> ConstraintData:
        """
        Data Transferable Object (DTO) of the class
        that implements this interface

        Returns
        ---------
        :obj:`ConstraintData`
            DTO of the object
        """
        return self._data

    @classmethod
    def from_data(cls, data: ConstraintData) -> Constraint:
        """
        Build an instance of the class that implements
        this interface from the proper DTO

        Parameters
        ---------
        :obj:`ConstraintData`
            DTO of the object

        Returns
        ---------
        :obj:`Constraint`
            The Constraint object built from DTO

        """
        from ..utils.utils import get_expr_from_data
        ineq_from_data = get_expr_from_data(data.ineq)
        return Constraint(id=data.id,
                          description=data.description,
                          ineq=ineq_from_data)
