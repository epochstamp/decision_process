from typing import Tuple, Union, List
from .expression.index_set import IndexSet,\
                                                             Index
from .variable import Variable
from .parameter import Parameter


class EvaluatorIndexSetError(BaseException):
    pass


class IndexSetDoesNotContainsIndexError(EvaluatorIndexSetError):
    pass

class IndexSequenceDoesNotExists(EvaluatorIndexSetError):
    pass

class LengthSeqMismatch(EvaluatorIndexSetError):
    pass


class ValueNotFoundError(BaseException):
    """
        Errors related to variable/parameter
        valuation issues
    """
    def __init__(self,
                 container: Union[Variable, Parameter],
                 idx_seq: Tuple[Index, ...],
                 inner_exception: BaseException):
        from .expression.indexed_container\
            import IndexedContainer
        idx_seq = [IndexSet(id=idx_seq[i].id,
                            parent=container.shape[i])
                   for i in range(len(idx_seq))]
        ic = IndexedContainer(container=container,
                              idx_seq=tuple(idx_seq))
        error = (str(ic) + " (detailed exception : "+str(inner_exception)
                         + " [" + str(type(inner_exception)) + "])")
        super(ValueNotFoundError, self).__init__(error)


class IndexSetNotFoundError(BaseException):
    """
        Errors related to index set
        valuation issues
    """
    def __init__(self,
                 idx: IndexSet,
                 inner_exception: BaseException):
        error = (idx.id +
                 " (detailed exception : " +
                 str(inner_exception)+")")
        super(IndexSetNotFoundError, self).__init__(error)


class Evaluator:
    """ Interface for evaluators
    """

    def _get(self,
             container: Union[Variable, Parameter],
             idx_seq: List[Index]):
        """
            Private method to override Evaluator.get behavior
            See Evaluator.get docstring
        """
        raise NotImplementedError()

    def _get_index(self, index_set: IndexSet) -> Index:
        """
            Private method to override Evaluator.get_index
            behavior
            See Evaluator.get_index docstring
        """
        raise NotImplementedError()

    def _get_all_components_by_index(self,
                                     index_set: IndexSet,
                                     indexes_source: List[Tuple[IndexSet, Index]] = []) -> List[Index]:
        """
            Private method to override
            Evaluator.get_all_components_by_index behavior
            /!\\ order not guaranteed for index set source, take care of symmetries
            See Evaluator.get_all_components_by_index docstring
        """
        raise NotImplementedError()

    def get(self,
            container: Union[Variable, Parameter],
            idx_seq: Tuple[Index, ...]):
        """Evaluates (possibly indexed) `container`
           (by `idx_seq`)

           Do not override this method. Override
           Evaluator._get instead

        Parameters
        -------
            container: :obj:`Variable` or :obj:`Parameter`
                       Symbolic container (variable, parameter).

            idx_seq: :obj:`list` of :obj:`Index`
                     List of indexes of the symbol.
                     Should match the shape of `symbol`.

        Returns
        -------
            An evaluation of (possibly indexed) `container` (with `idx_seq`).
            Output type is determined by the implemented object.

        Raises
        -------
        ValueNotFoundException
            If the indexed container cannot be evaluated

        """
        try:
            return self._get(container, idx_seq)
        except IndexSequenceDoesNotExists as e:
            raise e
        except BaseException as e:
            raise ValueNotFoundError(container, idx_seq, e)

    def get_index(self, index_set: IndexSet) -> Index:
        """Evaluate an index placeholder.

           Do not override this method. Override
           Evaluator._get_index instead

        Parameters
        -------
            name: :obj:`IndexSet`
                  Index Set.

        Returns
        -------
        :obj:`Index`
            A valuation of the index set

        Raises
        -------
        IndexNotFoundException
            If the indexed container cannot be evaluated

        """
        try:
            return self._get_index(index_set)
        except BaseException as e:
            raise IndexSetNotFoundError(index_set, e)

    def get_all_components_by_index(self,
                                    index_set: IndexSet,
                                    indexes_source: List[Tuple[IndexSet, Index]] = []) -> List[Index]:
        """Get all possible values of an index placeholder

           Do not override this method. Override
           Evaluator._get_all_components_by_index instead

           /!\\ order not guaranteed for index set source, take care of symmetries

        Parameters
        -------
            name: :obj:`str`
                  Name of the placeholder index.

        Returns
        -------
        :obj:`list` of `Index`
            List of indexes belonging to set `index_set`

        Raises
        -------
        IndexSetNotFoundException
            If the indexed container cannot be evaluated

        """
        indexes_source_buff = []
        for i in range(len(indexes_source)):
            index_set_source, index_source = indexes_source[i]
            if index_source not in self.get_all_components_by_index(index_set_source, indexes_source_buff):
                err = "Index " + str(index_source.id) + " does not belong "
                err += "to the index set " + str(index_set_source.id)
                if indexes_source_buff != []:
                    err += f"(mapped from the {indexes_source_buff} sequence of indexes)"
                raise IndexSetDoesNotContainsIndexError(err)
            indexes_source_buff += [indexes_source[i]]
        try:
            return self._get_all_components_by_index(index_set, indexes_source)
        except BaseException as e:
            raise IndexSetNotFoundError(index_set, e)
