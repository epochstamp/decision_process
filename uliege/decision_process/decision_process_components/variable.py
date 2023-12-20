from __future__ import annotations
from enum import Enum
import sys
import numpy as np
from typing import Tuple, Union
from pydantic import validator
from .expression.index_set import IndexSet
from .expression.container import Container,\
                                                  ContainerData


class NoVariableInvolvedError(BaseException):
    pass


class Type(Enum):
    REAL = (-np.inf, np.inf)
    NON_POSITIVE_REAL = (-np.inf, 0)
    NON_NEGATIVE_REAL = (0, np.inf)
    INTEGER = (-sys.maxsize, sys.maxsize)
    NON_POSITIVE_INTEGER = (-sys.maxsize, 0)
    NON_NEGATIVE_INTEGER = (0, sys.maxsize)
    BINARY = (0, 1)


DEFAULT_SUPPORT_BINARY = (0, 1)
DEFAULT_SUPPORT_INTEGER = (-sys.maxsize, sys.maxsize)
DEFAULT_SUPPORT_REAL = (-np.inf, np.inf)


def is_real(t: Type) -> bool:
    """
    Check that t is a real (sub)type

    Parameters
    ----------
    t : Type
        A Type Enum

    Returns
    ---------
    bool
        True if t is a real subtype
    """
    return t == Type.REAL or\
        t == Type.NON_NEGATIVE_REAL or\
        t == Type.NON_POSITIVE_REAL


def is_integer(t: Type) -> bool:
    """
    Check that t is an integer (sub)type

    Parameters
    ----------
    t : Type
        A Type Enum

    Returns
    ---------
    bool
        True if t is a integer subtype
    """
    return t == Type.INTEGER or\
        t == Type.NON_NEGATIVE_INTEGER or\
        t == Type.NON_POSITIVE_INTEGER


def is_binary(t: Type) -> bool:  # pragma: no cover
    """
    Check that t is an integer (sub)type

    Parameters
    ----------
    t : Type
        A Type Enum

    Returns
    ---------
    bool
        True if t is a binary type
    """
    return t == Type.BINARY


class VariableData(ContainerData):
    """`Variable` DTO
        See `Variable` for more details on attributes
    """
    v_type: Type = Type.REAL
    support: Union[Tuple[float, ...], Tuple[int, ...]] = tuple()

    @validator("support")
    @classmethod
    # Check consistency between support type and support variable
    def valid_support(cls, v, values, **kwargs):

        if values["v_type"] == Type.BINARY:
            return Type.BINARY.value

        if len(v) == 0:
            return values["v_type"].value

        if len(v) != 2:
            raise ValueError("`support` must contains exactly 2 values")

        if v[0] > v[1]:
            v = (v[1], v[0])

        if is_integer(values["v_type"]):
            v = tuple([(int(s) if s.is_integer() else s) for s in v])
            type_check = int
        elif is_real(values["v_type"]):
            type_check = float

        if not isinstance(v[0], type_check) or\
           not isinstance(v[1], type_check):
            raise TypeError("Mismatch between variable and support type")
        else:
            min_support_type = values["v_type"].value[0]
            max_support_type = values["v_type"].value[1]
            return (max(v[0], min_support_type),
                    min(v[1], max_support_type))


class Variable(Container):
    """ Decision process variable

        Attributes
        ----------
        id : str
            Id of the variable

        v_type : Type
            Variable type

        support : tuple of int or tuple of float
            Variable support

        shape: tuple of IndexSet
            Shape of the variable with index sets

    """

    def __init__(self,
                 id: str = "",
                 description: str = "",
                 v_type: Type = Type.REAL,
                 support: Union[Tuple[float, ...], Tuple[int, ...]] = tuple(),
                 shape: Union[IndexSet,
                              Tuple[IndexSet, ...]] = tuple()) -> None:
        super().__init__(id=id, shape=shape)
        self._data = VariableData(id=id,
                                  description=description,
                                  v_type=v_type,
                                  support=support,
                                  shape=shape)
        self._v_type = self._data.v_type
        self._support = self._data.support

    @property
    def v_type(self):
        return self._v_type

    @property
    def support(self):
        return self._support

    def get_data(self) -> VariableData:
        """
        Data Transferable Object (DTO) of `Variable` class

        Returns
        ---------
        VariableData
            DTO of the Variable object
        """
        return self._data

    @classmethod
    def from_data(cls, data: VariableData) -> Variable:
        """
        Build an instance of the class `Variable`
        from a `VariableData` DTO.

        Parameters
        ---------
        data : VariableData
            DTO of the `Variable` object

        Returns
        ---------
        Variable
            Instance of `Variable` built with `data`


        """
        return Variable(id=data.id,
                        description=data.description,
                        v_type=data.v_type,
                        support=data.support,
                        shape=data.shape)
