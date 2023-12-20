import pytest
from typing import Union, Tuple, List
import numpy as np
from uliege.decision_process import (
    Parameter,
    ParameterData
)
from uliege.decision_process import IndexedContainer
from uliege.decision_process import IndexSet
from hypothesis import given, assume
from hypothesis.strategies import builds,\
                                  text,\
                                  one_of,\
                                  lists
import random
from .utils import build_scalar_parameter

np.random.seed(1000)
random.seed(1000)

reasonable_lenghty_text = text(min_size=2, max_size=5)


@given(reasonable_lenghty_text,
       one_of(builds(IndexSet, id=reasonable_lenghty_text),
              lists(builds(IndexSet, id=reasonable_lenghty_text),
                    min_size=1,
                    max_size=3,
                    unique_by=lambda x: x.id)))
def test_valid_parameter(id: str,
                         shape: Union[IndexSet, Tuple[IndexSet, ...]]):
    """
    Test variable building with type consistent with support

    Parameters
    ----------
    id, shape:
        Attributes of the Parameter object
        (see Variable docstring)

    idx_seq: IndexSet or tuple of IndexSet
        A sequence of IndexSet

    Assumes
    ---------
        - Support values and Parameter share the same type

    Asserts
    ---------
        Successfully created a Parameter object

    """
    p = Parameter(id=id,
                  shape=shape)
    p_data = ParameterData(id=id,
                           shape=shape)
    assert p.get_data() == p_data


@given(reasonable_lenghty_text,
       builds(IndexSet, id=reasonable_lenghty_text),
       builds(IndexSet, id=reasonable_lenghty_text))
def test_valid_indexation_by_index_set(id: str,
                                       shape: IndexSet,
                                       idx_set: IndexSet):
    """
    Test parameter indexing where the indexing sequence length is
    equals to the shape length of the parameter

    Parameters
    ----------
    id, shape:
        Attributes of the Parameter object
        (see Parameter docstring)

    idx_seq: IndexSet or tuple of IndexSet
        A sequence of IndexSet

    Assumes
    ---------
        - Parameters are valid to build Parameter object
          (see Parameter docstring)
        - idx_seq length == shape length
        - for all i, idx_seq[i] == shape[i]

    Asserts
    ---------
        Successfully created an IndexedContainer

    """

    p = Parameter(id=id,
                  shape=shape)
    assume(shape == idx_set)
    indexed_var = IndexedContainer(container=p,
                                   idx_seq=idx_set)
    indexed_var_to_test = p[idx_set].get_data()
    assert(indexed_var_to_test == indexed_var.get_data())


@given(reasonable_lenghty_text,
       builds(IndexSet, id=reasonable_lenghty_text),
       builds(IndexSet,
              id=reasonable_lenghty_text,
              parent=builds(IndexSet,
                            id=reasonable_lenghty_text)))
def test_valid_indexation_by_index_subset(id: str,
                                          shape: IndexSet,
                                          idx_subset: IndexSet):
    """
    Test variable indexing where the indexing sequence length is
    equals to the shape length of the variable

    Parameters
    ----------
    id, shape:
        Attributes of the Variable object
        (see Variable docstring)

    idx_seq: IndexSet or tuple of IndexSet
        A sequence of IndexSet

    Assumes
    ---------
        - Parameters are valid to build Parameter object
          (see Variable docstring)
        - idx_seq inherits from shape

    Asserts
    ---------
        Successfully created an IndexedContainer

    """
    p = Parameter(id=id,
                  shape=shape)

    assume(idx_subset.parent == shape)
    indexed_par = IndexedContainer(container=p,
                                   idx_seq=(idx_subset,))
    indexed_par_to_test = p[idx_subset].get_data()
    assert(indexed_par_to_test == indexed_par.get_data())


@given(reasonable_lenghty_text,
       lists(builds(IndexSet,
                    id=reasonable_lenghty_text),
             min_size=1,
             max_size=3,
             unique_by=lambda x: x.id),
       lists(builds(IndexSet,
                    id=reasonable_lenghty_text),
             min_size=1,
             max_size=3,
             unique_by=lambda x: x.id))
def test_invalid_indexation_length_by_index_set(id: str,
                                                shape: Tuple[IndexSet, ...],
                                                idx_seq: Tuple[IndexSet, ...]):
    """
    Test parameter indexing where the indexing sequence length is
    either lower or higher than the shape of the parameter

    Parameters
    ----------
    id, shape:
        Attributes of the Parameter object
        (see Parameter docstring)

    idx_seq: IndexSet or tuple of IndexSet
        A sequence of IndexSet

    Assumes
    ---------
        - Parameters are valid to build Parameter object
          (see Parameter docstring)
        - idx_seq length =/= shape length

    Asserts
    ---------
        Failure by raising IndexError

    """
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
    p = Parameter(id=id,
                  shape=shape)
    with pytest.raises(IndexError) as e:
        _ = p[idx_seq]
    assert(e.type is IndexError)


@given(reasonable_lenghty_text,
       lists(builds(IndexSet, id=reasonable_lenghty_text),
             min_size=1,
             max_size=3,
             unique_by=lambda x: x.id),
       lists(builds(IndexSet, id=reasonable_lenghty_text),
             min_size=1,
             max_size=3,
             unique_by=lambda x: x.id))
def test_valid_indexation_by_index_set_sequence(id: str,
                                                shape: List[IndexSet],
                                                idx_set: List[IndexSet]):
    """
    Test variable indexing where the indexing sequence length is
    equals to the shape length of the parameter

    Parameters
    ----------
    id, support, shape:
        Attributes of the Parameter object
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
    idx_set = tuple(idx_set)
    shape = tuple(shape)
    v = Parameter(id=id,
                  shape=shape)
    assume(len(shape) == len(idx_set))
    assume(np.all([shape[i].id == idx_set[i].id
                   for i in range(len(shape))]))
    indexed_par = IndexedContainer(container=v,
                                   idx_seq=idx_set)
    indexed_par_to_test = v[idx_set].get_data()
    assert(indexed_par_to_test == indexed_par.get_data())


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
    id, shape:
        Attributes of the Variable object
        (see Variable docstring)

    idx_seq: IndexSet or tuple of IndexSet
        A sequence of IndexSet

    Assumes
    ---------
        - The two sequences are of the same length
        - It exists at least two differents index sets
          at the same location in both index and shape
          sequences

    Asserts
    ---------
        Failure by raising IndexError

    """
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
    v = Parameter(id=id,
                  shape=shape)
    with pytest.raises(IndexError) as e:
        _ = v[idx_seq]
    assert(e.type is IndexError)


@given(build_scalar_parameter())
def test_parameter_from_data(ic: IndexedContainer):
    p = ic.container
    p2 = Parameter.from_data(p.get_data())
    assert(p2.get_data() == p.get_data())
