from __future__ import annotations
from typing import Tuple
from .expression.index_set import IndexSet
from .expression.container import Container,\
                                                             ContainerData


class ParameterData(ContainerData):
    """`Parameter` DTO
    """


class Parameter(Container):
    """ Decision process parameter

        Attributes
        ----------

        v_type: Type
            Variable type

        support: tuple of int | tuple of float
            Variable support

        shape: tuple of IndexSet
            Shape of the variable with index sets

    """

    def __init__(self,
                 id: str = "",
                 description: str = "",
                 shape: Tuple[IndexSet, ...] = tuple()) -> None:
        Container.__init__(self,
                           id=id,
                           shape=shape,
                           description=description)
        self._data = ParameterData(id=id, shape=shape, description=description)

    def get_data(self) -> ParameterData:
        """
        Data Transferable Object (DTO) of Parameter class

        Returns
        ---------
        ParameterData
            DTO of a Parameter object
        """
        return self._data

    @classmethod
    def from_data(cls, data: ParameterData) -> Parameter:
        """
        Build an instance of the class `Parameter`
        from a `VariableData` DTO.

        Parameters
        ---------
        ParameterData
            DTO of a Parameter object

        Returns
        ---------
        Parameter
            A Parameter object


        """
        return Parameter(id=data.id,
                         description=data.description,
                         shape=data.shape)
