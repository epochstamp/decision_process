from typing import List, Any, Tuple, Union
from ..decision_process_components.variable import VariableData,\
                                                 is_integer,\
                                                 is_real
from ..decision_process_components.parameter import ParameterData
from ..decision_process_components.expression.container import ContainerData
from ..decision_process_components.expression.index_set import Index, IndexSet
from abc import ABC, abstractmethod
import numpy as np


class DatastreamError(BaseException):
    pass


class OutOfBoundsVarValueError(DatastreamError):
    pass


class ParameterNotLengthyEnough(DatastreamError):
    pass


class IndexSeqMismatch(DatastreamError):
    pass


class IndexSetDoesNotContainsIndexError(DatastreamError):
    pass


class DataError(DatastreamError):
    pass


class NotAFiniteNumber(DatastreamError):
    pass


class LengthSeqMismatch(DatastreamError):
    pass


class Datastream(ABC):
    """
        Interface for objects able to fetch
            - Variable initial values
            - Parameter vectors with an arbitrarily strictly positive length
            - Index sets components

        Be sure to inherits only the private methods
        when implementing this interface
        (security checks are performed around the calls)
    """

    @abstractmethod
    def _get_initialization(self,
                            var_data: VariableData,
                            idx_seq: List[Index]) -> Union[int,
                                                           float]:
        """
            Get initial value for any (possibly indexed) variable.
            Should return NaN if the list of indexes does not make sense
        """
        raise NotImplementedError()

    @abstractmethod
    def _get_parameter(self,
                       param_data: ParameterData,
                       idx_seq: List[Index],
                       length: int) -> Union[List[float],
                                             List[int],
                                             int,
                                             float]:
        """
            Get parameter vector of a given length
            for any (possibly indexed) variable.
            Should return NaN if the list of indexes does not make sense
        """
        raise NotImplementedError()

    @abstractmethod
    def _get_indexes_by_index_set(self,
                                  index_set: IndexSet,
                                  indexes_source: List[Tuple[IndexSet, Index]] = []) -> List[Index]:
        """
            Get indexes of a given index set and source indexes
            /!\\ order not guaranteed for index set source, take care of symmetries
        """
        raise NotImplementedError()

    def _compare_shape_and_idx_seq(self,
                                   container: ContainerData,
                                   idx_seq: List[Index]):
        """
            Ensure that
                - Index sequence and container lengths are equals
                - Each index belongs to the components of the corresponding
                  index set
        """
        shape = container.shape if isinstance(container.shape, tuple)\
            else (container.shape,)
        if len(shape) != len(idx_seq):
            cont_shape_len = str(len(shape))
            err = "Length of the container shape ("+cont_shape_len+")"
            err += " does not match the length ("+str(len(idx_seq))+")"
            err += " the index sequence ("+str(idx_seq)+")"
            raise IndexSeqMismatch(err)

        indexes_source_buff = []
        for i in range(len(shape)):
            idx = idx_seq[i]
            idx_set = shape[i]
            if idx.id != "neutral" and\
                    idx not in self.get_indexes_by_index_set(idx_set, indexes_source_buff):
                err = "Index " + str(idx.id) + " does not belong "
                err += "to the index set " + str(idx_set.id)
                if indexes_source_buff != []:
                    err += f"(mapped from the {indexes_source_buff} sequence of indexes)"
                raise IndexSetDoesNotContainsIndexError(err)
            indexes_source_buff += [(idx_set, idx)]

    def get_initialization(self,
                           var_data: VariableData,
                           idx_seq: List[Index]) -> Union[int,
                                                          float]:
        """
            Get initial value for any (possibly indexed) variable.

            Parameters
            ----------
            var_data: VariableData
                A Variable DTO

            idx_seq: list of Index
                List of indexes of the variable var_data

            Raises:
            TypeError
                If the value is not of the type specified
                by the variable

            OutOfBoundsVarValueErro
                If the value lies outside the support specified
                by the variable

            IndexSeqMismatch
                If the length of the index set sequence does not
                match the length of the variable shape

            IndexSetDoesNotContainsIndexError
                If any index in the index sequence does not belongs
                to its respective index set
        """
        self._compare_shape_and_idx_seq(var_data, idx_seq)
        indexed_by = (" indexed by " + str(idx_seq)) if\
            idx_seq != list() else ""
        try:
            value = self._get_initialization(var_data, idx_seq)
            if abs(value) <= 10e-8:
                value = 0
        except BaseException as e:
            err = "The following error occured while "
            err += "fetching the initialization of variable "
            err += var_data.id + indexed_by + ": "
            raise DataError(err) from e
        if not np.isfinite(value):
            err = "Initialization value of variable " + var_data.id
            err += indexed_by + " is not finite "
            err += "(value = " + str(value) + ")."
            raise NotAFiniteNumber(err)

        if is_integer(var_data.v_type) and isinstance(value, float)\
                and not value.is_integer():
            err = "The variable " + var_data.id + indexed_by + " is supposed "
            err += "to be of type integer but got real value instead "
            err += "(value = "+str(value)+")"
            raise TypeError(err)
        if value is not None and\
                (value < var_data.support[0] or value > var_data.support[1]):
            err = "The value " + str(value) + " is out of the bounds "
            err += str(var_data.support) + " for the variable "
            err += str(var_data.id) + indexed_by
            raise OutOfBoundsVarValueError(err)
        return value

    def get_parameter(self,
                      param_data: ParameterData,
                      idx_seq: List[Index],
                      length: int) -> Union[List[int],
                                            List[float]]:
        """
            Get parameter vector of a given length
            for any (possibly indexed) variable.

            Parameters
            ----------
            param_data: ParameterData
                A Parameter DTO

            idx_seq: list of Index
                List of indexes of the variable var_data

            length: int


            Raises:
            ValueError
                If the length is not strictly positive

            IndexSeqMismatch
                If the length of the index set sequence does not
                match the length of the variable shape

            IndexSetDoesNotContainsIndexError
                If any index in the index sequence does not belongs
                to its respective index set
        """
        self._compare_shape_and_idx_seq(param_data, idx_seq)
        indexed_by = (" indexed by " + str(idx_seq)) if\
            idx_seq != list() else ""
        if length < 1:
            raise ValueError("Length should be strictly positive")
        try:
            p = self._get_parameter(param_data, idx_seq, length)
        except BaseException as e:
            err = "The following error occured while "
            err += "fetching the value(s) of parameter "
            err += param_data.id + ": " + str(e)
            raise DataError(err) from e
        if not isinstance(p, list):
            p = [p]*length
        if p is not None and len(p) != length:
            err = "The values of parameter " + param_data.id
            err += " were requested for a horizon of " + str(length)
            err += " steps but got only " + str(len(p)) + " steps."
            raise ParameterNotLengthyEnough(err)

        if not np.all([np.isfinite(value) for value in p]):
            err = "Some value(s) of parameter " + param_data.id
            err += indexed_by + " is(are) not finite "
            err += "(value(s) = " + str(p) + ")."
            raise NotAFiniteNumber(err)

        return p

    def get_indexes_by_index_set(self,
                                 index_set: IndexSet,
                                 indexes_source: List[Tuple[IndexSet, Index]] = []) -> List[Index]:
        """
            Returns list of indexes given the sequence of indexes from their respective indexsets.

            Parameters
            ----------
            index_set: IndexSet
                An index set
            index_source: list of tuple of IndexSet, Index
                List of indexes who belongs to their respective index sets in index set mapping

            Returns
            ----------
            list of Index or None
                A list of index (possibly from index sets sources) if available, None otherwise
        """

        indexes_source_buff = []
        for i in range(len(indexes_source)):
            index_set_source, index_source = indexes_source[i]
            if index_source not in self.get_indexes_by_index_set(index_set_source, indexes_source_buff):
                err = "Index " + str(index_source.id) + " does not belong "
                err += "to the index set " + str(index_set_source.id)
                if indexes_source_buff != []:
                    err += f"(mapped from the {indexes_source_buff} sequence of indexes)"
                raise IndexSetDoesNotContainsIndexError(err)
            indexes_source_buff += [indexes_source[i]]
        try:
            return self._get_indexes_by_index_set(index_set, indexes_source)
        except BaseException as e:
            err = "The following error occured while "
            err += "fetching the indexes of index set "
            err += index_set.id + ": " + str(e)
            raise DataError(err)

    def activate_helper(self, var_data: VariableData, T: int) -> List[bool]:
        """
            Returns a boolean vector of length T which indicates whether a helper variable should be created each discrete time step between 0 and T-1
            By default, activate the variable at each time step
            Override it to disable variable at each desired time step. Only for computation time optimisation purpose.
            Parameters
            ----------
            T: int
                Length of the binary vector

            param_data: ParameterData
                A Variable DTO

            Returns
            ----------
            list of bool
                A list of flags to indicate whether the helper variable is active at each time step.
        """
        return [True]*T
