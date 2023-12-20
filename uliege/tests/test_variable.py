import pytest
from pydantic import ValidationError
from uliege.decision_process import IndexedContainer
from uliege.decision_process import (
    Variable,
    VariableData,
    Type,
    is_real,
    is_integer
)
from uliege.decision_process import IndexSet
from hypothesis import assume, given
from hypothesis.strategies import builds,\
                                  text,\
                                  floats,\
                                  sampled_from,\
                                  tuples,\
                                  one_of,\
                                  lists
import random
import numpy as np
from typing import Union, Tuple, List
from .utils import build_scalar_variable

np.random.seed(1000)
random.seed(1000)

reasonable_lenghty_text = text(min_size=2, max_size=5)


@given(reasonable_lenghty_text,
       sampled_from(Type),
       one_of(builds(IndexSet, id=reasonable_lenghty_text),
              lists(builds(IndexSet, id=reasonable_lenghty_text),
                    min_size=0,
                    max_size=1,
                    unique_by=lambda x: x.id)))
def test_valid_variable(id: str,
                        v_type: Type,
                        shape: Union[IndexSet, Tuple[IndexSet, ...]]):
    """
    Test variable building with type consistent with support

    Parameters
    ----------
    id, support, shape:
        Attributes of the Variable object
        (see Variable docstring)

    idx_seq: IndexSet or tuple of IndexSet
        A sequence of IndexSet

    Assumes
    ---------
        - Support values and Variable share the same type

    Asserts
    ---------
        Successfully created a Variable object

    """
    support = tuple()
    any_non_integers = np.any([(isinstance(v, float) and not v.is_integer())
                               for v in support])
    any_non_floats = np.any([not isinstance(v, float) or
                             (isinstance(v, float) and v.is_integer())
                             for v in support])
    consistent_integer = is_integer(v_type) and not any_non_integers
    consistent_real = is_real(v_type) and not any_non_floats
    assume(consistent_integer or consistent_real)
    v = Variable(id=id,
                 shape=shape,
                 v_type=v_type)
    v_data = VariableData(id=id,
                          shape=shape,
                          v_type=v_type,
                          support=support)
    assert v.get_data() == v_data


@given(reasonable_lenghty_text,
       lists(floats(),
             min_size=1,
             max_size=4))
def test_invalid_support_fail(id: str,
                              support: List[float]):
    """
    Test variable build on a support with an invalid length
    (different from 2)
    Expected to fail

    Parameters
    ----------
    id, support, shape:
        Attributes of the Variable object
        (see Variable docstring)

    Assumes
    ----------
        - len(support) != 2

    Asserts
    ---------
        Failed to create the variable

    """
    assume(len(support) != 2)
    support = tuple(support)
    message = "`support` must contains exactly 2 values"
    with pytest.raises(ValueError, match=message):
        _ = Variable(id=id,
                     v_type=Type.REAL,
                     support=support)


@given(reasonable_lenghty_text,
       sampled_from(Type),
       tuples(floats(), floats()),
       one_of(builds(IndexSet, id=text(min_size=2,
                                       max_size=3)),
              lists(builds(IndexSet, id=reasonable_lenghty_text),
                    min_size=0,
                    max_size=3,
                    unique_by=lambda x: x.id)))
def test_inconsistent_variable_type_with_support(id: str,
                                                 v_type: Type,
                                                 support: Tuple[float, ...],
                                                 shape: Union[
                                                        IndexSet,
                                                        Tuple[IndexSet, ...]]):
    """
    Test variable building with type different of the support
    values type

    Parameters
    ----------
    id, support, shape:
        Attributes of the Variable object
        (see Variable docstring)

    idx_seq: IndexSet or tuple of IndexSet
        A sequence of IndexSet

    Assumes
    ---------
        - Type of the variable is Integer
        - Non integer numbers are in support

    Asserts
    ---------
        Fails to create an Variable of type Integer

    """
    any_non_integers = np.any([not v.is_integer() for v in support])
    not_consistent_integer = is_integer(v_type) and any_non_integers
    assume(not_consistent_integer)
    with pytest.raises(ValidationError) as e:
        _ = Variable(id=id,
                     shape=shape,
                     v_type=v_type,
                     support=support)
    assert(e.type is ValidationError)


@given(reasonable_lenghty_text,
       tuples(floats(), floats()),
       builds(IndexSet, id=reasonable_lenghty_text),
       builds(IndexSet, id=reasonable_lenghty_text))
def test_valid_indexation_by_index_set(id: str,
                                       support: Tuple[float, ...],
                                       shape: IndexSet,
                                       idx_set: IndexSet):
    """
    Test variable indexing where the indexing sequence length is
    equals to the shape length of the variable

    Parameters
    ----------
    id, support, shape:
        Attributes of the Variable object
        (see Variable docstring)

    idx_seq: IndexSet or tuple of IndexSet
        A sequence of IndexSet

    Assumes
    ---------
        - Parameters are valid to build Variable object
          (see Variable docstring)
        - idx_seq length == shape length

    Asserts
    ---------
        Successfully created an IndexedContainer

    """
    v_type = Type.REAL
    v = Variable(id=id,
                 shape=shape,
                 v_type=v_type,
                 support=support)
    assume(shape == idx_set)
    indexed_var = IndexedContainer(container=v,
                                   idx_seq=(idx_set,))
    indexed_var_to_test = v[idx_set].get_data()
    assert(indexed_var_to_test == indexed_var.get_data())


@given(reasonable_lenghty_text,
       tuples(floats(), floats()),
       lists(builds(IndexSet, id=reasonable_lenghty_text),
             min_size=1,
             max_size=3,
             unique_by=lambda x: x.id),
       lists(builds(IndexSet, id=reasonable_lenghty_text),
             min_size=1,
             max_size=3,
             unique_by=lambda x: x.id))
def test_valid_indexation_by_index_set_sequence(id: str,
                                                support: Tuple[float, ...],
                                                shape: List[IndexSet],
                                                idx_set: List[IndexSet]):
    """
    Test variable indexing where the indexing sequence length is
    equals to the shape length of the variable

    Parameters
    ----------
    id, support, shape:
        Attributes of the Variable object
        (see Variable docstring)

    idx_set: IndexSet
        A IndexSet object

    Assumes
    ---------
        - Parameters are valid to build Variable object
          (see Variable docstring)
        - idx_set == shape

    Asserts
    ---------
        Successfully created an IndexedContainer

    """
    v_type = Type.REAL
    idx_set = tuple(idx_set)
    shape = tuple(shape)
    v = Variable(id=id,
                 shape=shape,
                 v_type=v_type,
                 support=support)
    assume(len(shape) == len(idx_set))
    assume(np.all([shape[i].id == idx_set[i].id
                   for i in range(len(shape))]))
    indexed_var = IndexedContainer(container=v,
                                   idx_seq=idx_set)
    indexed_var_to_test = v[idx_set].get_data()
    assert(indexed_var_to_test == indexed_var.get_data())


@given(reasonable_lenghty_text,
       tuples(floats(), floats()),
       builds(IndexSet, id=reasonable_lenghty_text),
       builds(IndexSet,
              id=reasonable_lenghty_text,
              parent=builds(IndexSet,
                            id=reasonable_lenghty_text)))
def test_valid_indexation_by_index_subset(id: str,
                                          support: Tuple[float, ...],
                                          shape: IndexSet,
                                          idx_subset: IndexSet):
    """
    Test variable indexing where the indexing sequence length is
    equals to the shape length of the variable

    Parameters
    ----------
    id, support, shape:
        Attributes of the Variable object
        (see Variable docstring)

    idx_seq: IndexSet or tuple of IndexSet
        A sequence of IndexSet

    Assumes
    ---------
        - Parameters are valid to build Variable object
          (see Variable docstring)
        - idx_subset inherits from shape

    Asserts
    ---------
        Successfully created an IndexedContainer

    """
    v_type = Type.REAL
    v = Variable(id=id,
                 shape=shape,
                 v_type=v_type,
                 support=support)

    assume(idx_subset.parent == shape)
    indexed_var = IndexedContainer(container=v,
                                   idx_seq=(shape,))
    indexed_var_to_test = v[shape].get_data()
    assert(indexed_var_to_test == indexed_var.get_data())


@given(reasonable_lenghty_text,
       tuples(floats(), floats()),
       lists(builds(IndexSet,
                    id=reasonable_lenghty_text),
             min_size=1,
             max_size=2,
             unique_by=lambda x: x.id),
       lists(builds(IndexSet,
                    id=reasonable_lenghty_text),
             min_size=1,
             max_size=2,
             unique_by=lambda x: x.id))
def test_invalid_indexation_length_by_index_set(id: str,
                                                support: Tuple[float, ...],
                                                shape: Tuple[IndexSet, ...],
                                                idx_seq: Tuple[IndexSet, ...]):
    """
    Test variable indexing where the indexing sequence length is
    either lower or higher than the shape of the variable

    Parameters
    ----------
    id, v_type, support, shape:
        Attributes of the Variable object
        (see Variable docstring)

    idx_seq: IndexSet or tuple of IndexSet
        A sequence of IndexSet

    Assumes
    ---------
        - Parameters are valid to build Variable object
          (see Variable docstring)
        - idx_seq length =/= shape length

    Asserts
    ---------
        Failure by raising IndexError

    """
    v_type = Type.REAL
    shape_is_idxset = isinstance(shape, IndexSet)
    if not shape_is_idxset:
        shape = tuple(shape)
    idxseq_is_idxset = isinstance(idx_seq, IndexSet)
    if not idxseq_is_idxset:
        idx_seq = tuple(idx_seq)

    both_idxseq = shape_is_idxset and idxseq_is_idxset
    if not both_idxseq:
        if shape_is_idxset:
            same_length = len(idx_seq) == 1
        elif idxseq_is_idxset:
            same_length = len(shape) == 1
        else:
            same_length = len(idx_seq) == len(shape)
    else:
        same_length = True

    assume(not same_length)
    v = Variable(id=id,
                 shape=shape,
                 v_type=v_type,
                 support=support)
    with pytest.raises(IndexError) as e:
        _ = v[idx_seq]
    assert(e.type is IndexError)


@given(reasonable_lenghty_text,
       one_of(builds(IndexSet, id=reasonable_lenghty_text),
              lists(builds(IndexSet, id=reasonable_lenghty_text),
                    min_size=1,
                    max_size=2,
                    unique_by=lambda x: x.id)),
       one_of(builds(IndexSet, id=reasonable_lenghty_text),
              lists(builds(IndexSet, id=reasonable_lenghty_text),
                    min_size=1,
                    max_size=2,
                    unique_by=lambda x: x.id)))
def test_invalid_indexation_setname_by_index_set(id: str,
                                                 shape: Union[IndexSet,
                                                              Tuple[IndexSet,
                                                                    ...]],
                                                 idx_seq: Union[IndexSet,
                                                                Tuple[IndexSet,
                                                                      ...]]):
    """
    Test variable indexing where the indexing sequence contains
    index sets that does neither match the shape name or inherits
    from an index that match the shape name

    Parameters
    ----------
    id, v_type, support, shape:
        Attributes of the Variable object
        (see Variable docstring)

    idx_seq: IndexSet or tuple of IndexSet
        A sequence of IndexSet

    Assumes
    ---------
        - The two sequences are of the same length
        - It exists at least two differents index sets
          at the same location in both index and shape
          sequences.

    Asserts
    ---------
        Failure by raising IndexError

    """
    v_type = Type.REAL
    shape_is_idxset = isinstance(shape, IndexSet)
    if not shape_is_idxset:
        shape = tuple(shape)
    idxseq_is_idxset = isinstance(idx_seq, IndexSet)
    if not idxseq_is_idxset:
        idx_seq = tuple(idx_seq)

    both_idxseq = shape_is_idxset and idxseq_is_idxset
    if not both_idxseq:
        if shape_is_idxset:
            same_length = len(idx_seq) == 1
        elif idxseq_is_idxset:
            same_length = len(shape) == 1
        else:
            same_length = len(idx_seq) == len(shape)
    else:
        same_length = True

    assume(same_length)
    shape_t = tuple(shape) if not shape_is_idxset else (shape,)
    idxseq_t = tuple(idx_seq) if not idxseq_is_idxset else (idx_seq,)
    truth_sequence = [not idxseq_t[i] == shape_t[i]
                      for i in range(len(shape_t))]
    two_different_shape_index_set = np.any(truth_sequence)
    assume(two_different_shape_index_set)
    v = Variable(id=id,
                 shape=shape,
                 v_type=v_type)
    with pytest.raises(IndexError) as e:
        _ = v[idx_seq]
    assert(e.type is IndexError)


@given(build_scalar_variable())
def test_variable_from_data(ic: IndexedContainer):
    v = ic.container
    v2 = Variable.from_data(v.get_data())
    assert(v2.get_data() == v.get_data())
