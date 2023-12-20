from __future__ import annotations
from ...decision_process_components.expression.numeric_expression\
    import NumericExpression, NumericExpressionData
from typing import Tuple, Union
from ...decision_process_components.expression.index_set import IndexSet
from ...decision_process_components.variable import Variable, VariableData
from ...decision_process_components.parameter import Parameter, ParameterData
from ...decision_process_components.evaluator import Evaluator
import numpy as np


class IndexedContainerData(NumericExpressionData):
    idx_seq: Tuple[IndexSet, ...] = tuple()


class IndexedVariableData(IndexedContainerData):
    """ `IndexedContainer` DTO
         See `IndexedContainer` for more details on attributes
    """
    from ...decision_process_components.variable import Variable

    container: VariableData


class IndexedParameterData(IndexedContainerData):
    """ `IndexedContainer` DTO
         See `IndexedContainer` for more details on attributes
    """
    from ...decision_process_components.parameter import Parameter

    container: ParameterData


class IndexedContainer(NumericExpression):

    """ Container with mixed sequence of index sets and indexes

        The two latters are used for reducing an expression
        over a index set with indexes

        Attributes
        ----------
        container: Variable or Parameter
            The container, either a variable or a parameter

        idx_seq: :obj:`tuple` of :obj:`IndexSet` (optional)
            Index set sequence

        Raises
        ----------
        IndexError
            If the length of index sequence does not match
            or any index set does not match or inherits
            of the corresponding shape

    """

    def __init__(self,
                 container: Union[Variable, Parameter],
                 idx_seq: Union[IndexSet, Tuple[IndexSet, ...]] = tuple()):
        from ...utils import utils
        idx_seq = ((idx_seq,) if isinstance(idx_seq, IndexSet)
                   else tuple(idx_seq))
        shape = container.shape
        shape = (shape,) if isinstance(shape, IndexSet) else tuple(shape)
        if len(idx_seq) != len(shape):
            error = "Must be indexed by as many components as in `shape`"
            raise IndexError(error)
        lst_has_parent = [utils.has_parent(idx_seq[i], shape[i])
                          for i in range(len(shape))]
        if not np.all(lst_has_parent):
            error = "At least one of the index set does not match "
            error += "with or inherits from the corresponding shape"
            raise IndexError(error)

        if isinstance(container, Variable):
            self._data = IndexedVariableData(container=container.get_data(),
                                             idx_seq=idx_seq)
        else:
            self._data = IndexedParameterData(container=container.get_data(),
                                              idx_seq=idx_seq)
        self._container = container

    @property
    def container(self) -> Union[Variable, Parameter]:
        """ The container
        """
        return self._container

    def __str__(self):
        container_str = self.container.id
        for idx_set in self._data.idx_seq:
            container_str += "[" + idx_set.id + "]"
        return container_str

    def __call__(self, evaluator: Evaluator):
        idx_eval_seq = list()
        for idx_set in self.get_data().idx_seq:
            idx_eval_seq.append(evaluator.get_index(idx_set))
        return evaluator.get(container=self.container,
                             idx_seq=tuple(idx_eval_seq))

    def get_data(self) -> Union[IndexedVariableData, IndexedParameterData]:
        """
        Data Transferable Object (DTO) of `IndexedContainer` class

        Returns
        ---------
        :obj:`IndexedContainerData`
            DTO of the `IndexedContainer` object
        """
        return self._data

    @classmethod
    def from_data(cls, data: Union[IndexedVariableData,
                                   IndexedParameterData])\
            -> Union[IndexedVariableData, IndexedParameterData]:
        """
        Build an instance of the class `IndexedContainer`
        from a `IndexedContainerData` DTO.

        Parameters
        ---------
        :obj:`VariableData`
            DTO of the `IndexedContainer` object

        Returns
        ---------
        :obj:`Variable`
            Instance of `IndexedContainerData` built with `data`


        """
        if isinstance(data.container, VariableData):
            ic_cls = Variable
        else:
            ic_cls = Parameter
        container_from_data = ic_cls.from_data(data.container)
        return IndexedContainer(container_from_data,
                                data.idx_seq)

    def __repr__(self):
        idx_seq = ",".join([idx_set.id for idx_set in self._data.idx_seq])
        return f"{self._container.id}[({idx_seq})]"
