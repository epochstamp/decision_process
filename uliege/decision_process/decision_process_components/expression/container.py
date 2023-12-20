from __future__ import annotations
from typing import Tuple, Union
from ...decision_process_components.expression.index_set import IndexSet
from ...decision_process_components.expression.operand import Operand
from ...base.base_component import BaseComponent, BaseComponentData
from pydantic import validator


class UnicityError(BaseException):
    pass


class ContainerData(BaseComponentData):
    """ `Container` DTO
         See `Container` for more details on attributes
    """
    shape: Union[Tuple[IndexSet, ...], IndexSet] = tuple()


class Container(BaseComponent, Operand):

    """Base class for decision process variables and parameters.

       Should not be used directly, only for inheritance purpose.

       Attributes
       ----------
       id: :obj:`str`
           Id of the variable

       shape: :obj:`tuple` of :obj:`IndexSet`
           Shape of the variable with index sets

    """

    def __init__(self,
                 id: str = "",
                 shape: Union[IndexSet, Tuple[IndexSet, ...]] = tuple(),
                 description: str = "")\
            -> None:
        BaseComponent.__init__(self, id=id, description=description)
        self._shape = shape
        self._data = ContainerData(id=id,
                                   shape=shape,
                                   description=description)

    @property
    def shape(self):
        """ Container shape
        """
        return self._shape

    def __getitem__(self,
                    idx_seq: Union[IndexSet,
                                   Tuple[IndexSet, ...]])\
            -> IndexedContainer:
        """Returns an indexed container with a mixed sequence
           of index sets, indexes and reduce index sets

           Parameters
           ----------
           `container`: :obj:`Container`
                The container to be indexed

           `idx_seq`: :obj:`IndexSet`
                      | :obj:`Index`
                      | :obj:`tuple`
                      of (:obj:`IndexSet` | :obj:`Index`)

            Returns
            ---------
            :obj:`IndexedContainer`
                The indexed container with `idx_seq`

        """
        from ...decision_process_components.\
            expression.indexed_container import\
            IndexedContainer
        idx_container = IndexedContainer(container=self,
                                         idx_seq=idx_seq)
        return idx_container
