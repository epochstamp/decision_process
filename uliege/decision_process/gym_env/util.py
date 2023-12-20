from typing import List, Union, Dict, Callable
from ..decision_process_components.variable import Variable, is_real
from ..decision_process_components.parameter import Parameter
from ..decision_process_components.expression.index_set import IndexSet, Index
import numpy as np
from gym.spaces import Box, Dict as SpaceDict
import itertools


def get_dict_space_from_container_set(
    containers: List[Union[Variable,
                           Parameter]],
    components_by_index_set_func: Callable[[IndexSet],
                                           List[Index]])\
        -> SpaceDict:
    """
        Create a gym SpaceDict from a list of variable/parameters

        Parameters
        ----------
        containers: list of Variable or Parameter objects
            The containers
        components_by_index_set_funcs:
            callable with IndexSet as input and
            list of Index as output
            A function mapping IndexSet objects to list of Index objects

        Returns
        -----------
        SpaceDict
            A SpaceDict object used to specify observation/action spaces


    """
    dict_space = dict()
    for c in containers:
        cshape = c.shape if\
            isinstance(c.shape, tuple) else (c.shape,) 
        if isinstance(c, Variable):
            dtype = np.float64 if is_real(c.get_data().v_type) else np.int32
            c_space = Box(low=np.asarray(c.get_data().support[0]),
                          high=np.asarray(c.get_data().support[1]),
                          dtype=dtype)
        else:
            c_space = Box(low=np.asarray(-np.inf), high=np.asarray(np.inf), dtype=np.float64)
        if cshape == tuple():
            dict_space[c.id, tuple()] = c_space
        else:
            index_set_comps = {i: components_by_index_set_func(i)
                               for i in cshape}
            index_set_vals = (dict(zip(index_set_comps, x))
                              for x in itertools.product(
                                 *index_set_comps.values()))
            for index_set_val in index_set_vals:
                dict_space[c.id, tuple([index_set_val[i]
                                       for i in cshape])] = c_space
    return SpaceDict(dict_space)
