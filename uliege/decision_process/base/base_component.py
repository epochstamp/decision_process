
from pydantic import BaseModel, validator
from .transferable import Transferable
from uuid import uuid4


class BaseComponentData(BaseModel):
    """ DTO of `BaseComponent`
        See `BaseComponent` docstring for more details

    """

    id: str = ""
    description: str = ""

    @validator("id")
    @classmethod
    def uuid_if_empty_id(cls, v):
        if v == "":
            return str(uuid4())
        else:
            return v


class BaseComponent(Transferable):

    """ Base class for all objects
        uniquely identified by a user-defined ID

        Attributes
        -----------
        id: str
            The ID of the hashable object

    """

    def __init__(self, id: str = "", description = ""):
        if id == "":
            id = str(uuid4())
        self._data = BaseComponentData(id=id, description=description)

    @property
    def id(self): return self._data.id

    @property
    def description(self): return self._data.description

    def __hash__(self):
        return hash(type(self)) + hash(self.id)
