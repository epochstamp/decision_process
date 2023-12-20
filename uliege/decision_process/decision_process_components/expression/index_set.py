from __future__ import annotations
from pydantic.dataclasses import dataclass
from dataclasses import field
from typing import List, Type, Union
from uuid import uuid4


@dataclass(frozen=True, order=True)
class Index:
    """Index element. Used in `ReduceIndexSet` objects.

       Is directly a Data Transferable Object (DTO)

       Parameters
       ----------
       name: :obj:`str`
            Name of the index
    """
    id: str


@dataclass(frozen=True, order=True)
class IndexSet:
    """Index set of any container (e.g., parameter, variable...)
       Is directly a Data Transferable Object (DTO)

       Parameters
       ----------
       set_name: :obj:`str`
            Name of the index set

       parent: IndexSet
            The index set which is a superset of this one
    """
    id: str = str(uuid4())
    description: str = field(hash=False, compare=False, default="")
    parent: Union[IndexSet, Type[None]] = field(hash=False, compare=False, default=None)


dataclass(IndexSet).__pydantic_model__.update_forward_refs()
