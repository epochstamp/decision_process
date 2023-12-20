# Determine version number if it exits (i.e. if package is installed)
try:
    from uliege.decision_process import _version

    __version__ = _version.version
except ImportError:
    # package is not installed
    pass


from .decision_process import (
    DecisionProcess,
    DecisionProcessError,
    DecisionProcessData,
    DecisionProcessRealisation,
    ContainerSequence,
    UselessVariableError,
)
from .controllers import ForwardController, TrajectoryOptimizer
from .controllers.optim_forward_controller import OptimForwardController
from .datastream import (
    Datastream,
    DatastreamFactory,
    ParameterNotLengthyEnough,
    DataError,
    DatastreamOverrider,
)
from .decision_process_simulator.decision_process_simulator import (
    DecisionProcessSimulator,
)
from .decision_process_components import (
    Parameter,
    ParameterData,
    Variable,
    VariableData,
    Dynamics,
    Constraint,
    CostFunction,
    sum_reduce,
    prod_reduce,
    Type,
    LastTimeStepMask,
    UniformStepMask,
    HorizonMask,
    Reduce,
    Reducer,
    ReduceData,
    ReduceIndexSet,
    IndexSet,
    Index,
    is_real,
    is_binary,
    is_integer,
    NumericExpressionData,
    NumericExpression,
    LogicalExpressionData,
    LogicalExpression,
    IndexedContainerData,
    IndexedContainer,
    BinopData,
    Binop,
    Binoperator,
    Unop,
    UnopData,
    Unoperator,
    Ineqoperator,
    NoVariableInvolvedError,
    Container,
    ContainerData,
    Evaluator,
)
from .utils.utils import (
    get_variables,
    free_index_sets,
    reduce_index_sets,
    get_parameters,
    localize_dict_by_key,
    extract_all_dict_leaves,
)
from .base.base_component import BaseComponent
from .base.transferable import Transferable
