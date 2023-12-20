from pydantic import BaseModel
from typing import Any
from abc import ABC, abstractmethod


class Transferable(ABC):
    """Interface for objects that can produce and be built from DTOs"""

    @abstractmethod
    def get_data(self) -> BaseModel:
        """
        Data Transferable Object (DTO) of the class
        that implements this interface

        Returns
        ---------
        :obj:`BaseModel`
            DTO of the object
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def from_data(cls, data: BaseModel) -> Any:
        """
        Build an instance of the class that implements
        this interface from the proper DTO

        Parameters
        ---------
        :obj:`BaseModel`
            DTO of the object

        Returns
        ---------
        :obj:


        """
        raise NotImplementedError()
