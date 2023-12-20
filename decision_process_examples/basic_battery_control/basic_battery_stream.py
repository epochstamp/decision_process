# -*- coding: UTF-8 -*-

from typing import List, Union
from uliege.decision_process import BaseComponent
from uliege.decision_process import (
    extract_all_dict_leaves,
    localize_dict_by_key
)
from uliege.decision_process import VariableData
from uliege.decision_process import ParameterData
from uliege.decision_process import Index, IndexSet
from uliege.decision_process import Datastream


# Should be abstracted by containing a repository
class BasicBatteryStream(Datastream):

    def __init__(self, current_time=0):
        self._current_time = 0

        mega_db = dict()
        mega_db = dict()
        mega_db["parameters"] = dict()
        mega_db["parameters"]["charge_efficiency"] = dict()
        mega_db["parameters"]["charge_efficiency"]["perfect_storage"] = 1.0
        mega_db["parameters"]["charge_efficiency"]["rusty_storage"] = 0.3
        mega_db["parameters"]["discharge_efficiency"] = dict()
        mega_db["parameters"]["discharge_efficiency"]["perfect_storage"] = 1.0
        mega_db["parameters"]["discharge_efficiency"]["rusty_storage"] = 0.3
        mega_db["parameters"]["max_charge_power"] = dict()
        mega_db["parameters"]["max_charge_power"]["perfect_storage"] = 100
        mega_db["parameters"]["max_charge_power"]["rusty_storage"] = 10
        mega_db["parameters"]["max_discharge_power"] = dict()
        mega_db["parameters"]["max_discharge_power"]["perfect_storage"] = 100
        mega_db["parameters"]["max_discharge_power"]["rusty_storage"] = 10
        mega_db["parameters"]["max_storage_capacity"] = dict()
        mega_db["parameters"]["max_storage_capacity"]["perfect_storage"] = 1000
        mega_db["parameters"]["max_storage_capacity"]["rusty_storage"] = 100
        mega_db["parameters"]["state_of_charge_val"] = dict()
        mega_db["parameters"]["state_of_charge_val"]["perfect_storage"] = 10
        mega_db["parameters"]["state_of_charge_val"]["rusty_storage"] = 0.1
        mega_db["parameters"]["max_cumulative_power"] = 1000000
        mega_db["parameters"]["delta_time"] = 0.1

        mega_db["initialization"] = dict()
        mega_db["initialization"]["state_of_charge"] = dict()
        mega_db["initialization"]["state_of_charge"]["perfect_storage"] = 0
        mega_db["initialization"]["state_of_charge"]["rusty_storage"] = 0
        mega_db["initialization"]["cumulative_charge"] = 0

        mega_db["indexes"] = dict()
        mega_db["indexes"]["batteries"] = dict()
        batteries = ["rusty_storage", "perfect_storage"]
        mega_db["indexes"]["batteries"]["linear_batteries"] = batteries

        self.mega_db = mega_db

    def _subdb_query(self, subdb, path: List[str]) -> Union[int, float, None]:
        d = subdb
        for p in path:
            d = d.get(p, None)
            if d is None:
                return None
        return d

    def _get_initialization(self,
                            var_data: VariableData,
                            idx_seq: List[Index]) -> Union[int,
                                                           float,
                                                           None]:
        path = [var_data.id] + [i.id for i in idx_seq]
        return self._subdb_query(self.mega_db["initialization"], path)

    def _get_parameter(self,
                       param_data: ParameterData,
                       idx_seq: List[Index],
                       length: int) -> Union[int,
                                             float,
                                             None]:
        path = [param_data.id] + [i.id for i in idx_seq]
        data = self._subdb_query(self.mega_db["parameters"], path)
        if not isinstance(data, list):
            data = [data]*length
        else:
            data = data[self._current_time: self._current_time + length]
        return data

    def _get_indexes_by_index_set(self,
                                  index_set: IndexSet) -> Union[None,
                                                                List[Index]]:
        dict_indexes = localize_dict_by_key(self.mega_db["indexes"],
                                            index_set.id)
        indexes = extract_all_dict_leaves(dict_indexes)
        indexes = [item for sublist in indexes for item in sublist]
        indexes = [Index(id=i) for i in indexes]
        return (indexes if indexes != [] else None)
