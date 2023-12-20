from pylatex import Document, Section, Subsection, LongTable,\
    Math, TikZ, Axis, Plot, Figure, Package, Alignat, VerticalSpace, NewPage
from uliege.decision_process import DecisionProcessData
from uliege.decision_process import (
    free_index_sets,
    reduce_index_sets,
    get_variables,
    get_parameters
)
from uliege.decision_process import is_real, is_integer, is_binary, Type
from operator import add, sub, mul, truediv, neg, abs
from uliege.decision_process import Reducer, ReduceData
from uliege.decision_process import NumericExpressionData
from uliege.decision_process import IndexedContainerData
from uliege.decision_process import BinopData
from uliege.decision_process import UnopData
from uliege.decision_process import Ineqoperator
from uliege.decision_process import IndexSet
from uliege.decision_process import (
    LastTimeStepMask,
    UniformStepMask,
    HorizonMask
)
import numpy as np
from typing import Callable, Union, Tuple


def type_to_str(t: Type) -> str:
    """
        Stringify a Type enum

        Arguments
        ---------
        t: Type
            a Type enum

        Returns
        ---------
        str
            A string representation, compatible with LaTeX
    """
    if is_real(t):
        return r"\mathbb{R}"
    elif is_integer(t):
        return r"\mathbb{N}"
    elif is_binary(t):
        return r"\mathbb{B}"
    else:
        return ""


def comparison_to_str(ineq_op: Ineqoperator) -> str:
    """
        Stringify an inequality operator

        Arguments
        ---------
        ineq_op: Ineqoperator
            an inequation operator

        Returns
        ---------
        str
            A string representation, compatible with LaTeX
    """
    return r"\leqslant " if ineq_op == Ineqoperator.LE else\
        (r"\geqslant " if ineq_op == Ineqoperator.GE else
         ("=" if ineq_op == Ineqoperator.EQ else ""))


def operation_to_str(expr_1_str: str,
                     operator: Callable,
                     expr_2_str: str = "",
                     put_parenthesis_left: bool = False,
                     put_parenthesis_right: bool = False) -> str:
    """
        Stringify a numeric operation

        Arguments
        ---------
        expr_1_str: str
            A stringifyed operation
        operator: Callable
            An arithmetic callable operator
        expr_2_str: str (optional, default="")
            Another stringifyed operation (binary ops)
        put_parenthesis_left: bool (optional, default=False)
            Whether to put parenthesis around the first expression
        put_parenthesis_right: bool (optional, default=False)
            Whether to put parenthesis around the second expression

        Returns
        ---------
        str
            A string representation, compatible with LaTeX
    """
    if put_parenthesis_left:
        expr_1_str = r"\left(" + expr_1_str + r"\right)"
    if put_parenthesis_right:
        expr_2_str = r"\left(" + expr_2_str + r"\right)"
    if operator == add:
        return expr_1_str + " + " + expr_2_str
    elif operator == mul:
        return expr_1_str + expr_2_str
    elif operator == sub:
        return expr_1_str + " - " + expr_2_str
    elif operator == truediv:
        return r'\frac{' + expr_1_str + '}{' + expr_2_str + '}'
    elif operator == neg:
        return "-" + expr_1_str
    elif operator == abs:
        return "|" + expr_1_str + "|"


def numerical_expression_to_str(expr: NumericExpressionData,
                                tdiff: int = 0,
                                idx_set_mapping=dict(),
                                cont_id_mapping=dict(),
                                sub_idxset_id_mapping=dict()):
    """
        Stringify a numerical expression

        Arguments
        ---------
        expr: str
            A numerical expression DTO
        tdiff: int
            Integer shift of the time step in the decision process
        idx_set_mapping: dict
            Mapping between ids and strings for index sets
        cont_id_mapping: dict
            Mapping between ids and strings
            for containers(parameter/variable)
        sub_idxset_id_mapping: dict
            Mapping between ids and strings
            for index sets inside reduce operators

        Returns
        ---------
        str
            A string representation, compatible with LaTeX
    """
    if isinstance(expr, IndexedContainerData):
        # Display id with the indexes in superscript and time index
        # in subscript with time diff
        container = expr
        idx_seq = [(idx_set_mapping.get(i.id, i.id)
                   if i.id not in sub_idxset_id_mapping
                   else sub_idxset_id_mapping.get(i.id, i.id))
                   for i in container.idx_seq]
        tdisplay =\
            ("t-" + str(tdiff)) if tdiff < 0\
            else ("t+" + str(tdiff) if tdiff > 0 else "t")
        indexes =\
            (", ".join(idx_seq) + ",")\
            if len(idx_seq) > 0 else ""
        cont_id = ("{" + cont_id_mapping.get(container.container.id,
                                            container.container.id) + "}").replace("^", "_{%temp}^", 1)
        if cont_id.count("_{%temp}^") == 0:
            return cont_id + "_{" + indexes + str(tdisplay) + "}"
        else:
            return cont_id.replace("%temp", indexes + str(tdisplay))
    elif isinstance(expr, BinopData):
        # Recursive display of a binary operation
        expr_1_str = numerical_expression_to_str(expr.expr_1,
                                                 tdiff,
                                                 idx_set_mapping,
                                                 cont_id_mapping,
                                                 sub_idxset_id_mapping)
        expr_2_str = numerical_expression_to_str(expr.expr_2,
                                                 tdiff,
                                                 idx_set_mapping,
                                                 cont_id_mapping,
                                                 sub_idxset_id_mapping)
        left_par = False
        right_par = False
        if expr.binop.value == mul:
            if not isinstance(expr.expr_1, IndexedContainerData)\
                    and not isinstance(expr.expr_1, ReduceData):
                left_par = True
            if not isinstance(expr.expr_2, IndexedContainerData)\
                    and not isinstance(expr.expr_2, ReduceData):
                right_par = True
        return operation_to_str(expr_1_str,
                                expr.binop.value,
                                expr_2_str,
                                left_par,
                                right_par)
    elif isinstance(expr, UnopData):
        # Recursive display of an unary operation
        expr_str = numerical_expression_to_str(expr.expr,
                                               tdiff,
                                               idx_set_mapping,
                                               cont_id_mapping,
                                               sub_idxset_id_mapping)
        return operation_to_str(expr_str, expr.unop.value)
    elif isinstance(expr, ReduceData):
        # Recursive display of a reduce operation (sum, prod...)
        reducer = expr
        expr = reducer.inner_expr
        idx_set = reducer.idx_reduce_set.idx_set
        idx_substit = sub_idxset_id_mapping.get(idx_set.id, idx_set.id[0].lower())
        idx_set_alias = idx_set_mapping.get(idx_set.id, idx_set.id)
        while idx_substit+"'" in sub_idxset_id_mapping.values():
            idx_substit += "'"
        sub_idxset_id_mapping_copy = dict(sub_idxset_id_mapping)
        sub_idxset_id_mapping_copy.update({idx_set.id: idx_substit})
        expr_str = numerical_expression_to_str(expr,
                                               tdiff,
                                               idx_set_mapping,
                                               cont_id_mapping,
                                               sub_idxset_id_mapping_copy)
        
        if reducer.idx_reduce_set.reducer == Reducer.SUM:
            return (r'\sum_{' + idx_substit
                    + r' \in ' + idx_set_alias + '}{' + expr_str + '}')
        elif reducer.idx_reduce_set.reducer == Reducer.PROD:
            return (r'\product_{' + idx_substit
                    + r' \in ' + idx_set_alias + '}{' + expr_str + '}')
    else:
        raise NotImplementedError(expr.__class__.__name__)


def support_to_str(v_type: Type,
                   support: Tuple[Union[float, int],
                                  Union[float, int]]) -> str:
    """
        Stringify a support

        Parameters
        ----------
        support: tuple of int/float and int/float
            A support interval

        Returns
        --------
        str
            A string representation, compatible with LaTeX
    """
    lbr = r"\left[" if is_real(v_type) else r"\left\{"
    rbr = r"\right]" if is_real(v_type) else r"\right\}"
    ldots = r"\ldots" if not is_real(v_type) else ""
    low = r"\infty" if support[0] == np.inf else\
          (r"-\infty" if support[0] == -np.inf else str(support[0]))
    high = r"\infty" if support[1] == np.inf else\
           (r"-\infty" if support[1] == -np.inf else str(support[1]))
    lst = [low, ldots, high] if ldots != "" else [low, high]
    return lbr + ",".join(lst) + rbr


def temporal_mask_to_str(temporal_mask: HorizonMask) -> str:
    """
        Stringify a temporal mask

        Parameters
        ----------
        support: HorizonMask
            A time-horizon mask

        Returns
        --------
        str
            A string representation, compatible with LaTeX
    """
    if isinstance(temporal_mask, UniformStepMask):
        return "always 1"
    elif isinstance(temporal_mask, LastTimeStepMask):
        return "0 if t < T else 1"


def decision_process_to_pdf(decision_process_data: DecisionProcessData,
                            path="./",
                            cont_id_mapping=dict(),
                            idxset_id_mapping=dict(),
                            func_id_mapping=dict(),
                            sub_idxset_id_mapping=dict(),
                            get_tex=False) -> None:
    """
        Create PDF and LaTeX documents from a decision process object

        Parameters
        -----------
        decision_process_data: DecisionProcessData
            A DecisionProcess DTO
        path:
            The root path to dump the documents (optional, default="./")
            
        cont_id_mapping: dict of str to str (optional, default=dict())
            Mapping between container ids and strings
        func_id_mapping: dict of str to str (optional, default=dict())
            Mapping between function ids and strings
        get_tex: bool (optional, default=False)
            Whether tex file is also created with the same name as pdf
            (id of the decision process)

    """
    doc = Document()
    doc.packages.append(Package('amssymb'))
    doc.packages.append(Package('mathtools'))
    with doc.create(Section(decision_process_data.id)):
        doc.append('This section describes the whole mathematical\
            model behind the abovementioned sequential decision process.')

        with doc.create(Subsection("Description")):
            doc.append(decision_process_data.description)
        with doc.create(Subsection('Index sets')):
            # Display the index sets in a table
            # that reports the id and the description
            # of each index set
            # Fetch index sets from the whole set of variables
            # and parameters
            variables = decision_process_data.state_variables\
                + decision_process_data.action_variables\
                + decision_process_data.helper_variables
            contdatas = variables + decision_process_data.parameters
            index_sets = {i for contdata in contdatas
                          for i in
                          ((contdata.shape,)
                           if isinstance(contdata.shape, IndexSet)
                           else contdata.shape)}

            # Fetch index sets from the whole set of dynamics,
            # constraints, and cost functions
            for dynamics in decision_process_data.dynamics_functions:
                index_sets =\
                    index_sets.union(
                        free_index_sets(dynamics.state_var_update).union(
                            reduce_index_sets(dynamics.state_var_update)
                        )
                    )
                index_sets = index_sets.union(dynamics.state_var.idx_seq)

            for cost_function in decision_process_data.cost_functions:
                index_sets =\
                    index_sets.union(
                        free_index_sets(cost_function.cost_expression).union(
                            reduce_index_sets(cost_function.cost_expression)
                        )
                    )

            for constraint in decision_process_data.constraint_functions:
                index_sets =\
                    index_sets.union(
                        free_index_sets(constraint.ineq).union(
                            reduce_index_sets(constraint.ineq)
                        )
                    )
            # Report the index set informations (id, descr, ID) in a table
            with doc.create(LongTable('|c|c|c|')) as data_table:
                data_table.add_hline()
                data_table.add_row(["Index set ID", "Description", "ID"])
                data_table.add_hline()
                for idx_set in index_sets:
                    data_table.add_row(
                        [idx_set.id,
                         idx_set.description,
                         Math(data=idxset_id_mapping.get(idx_set.id,
                                                         idx_set.id),
                              inline=True,
                              escape=False)])
                    data_table.add_hline()
                data_table.add_hline()

            # Report the parameter informations
            # (shape, description, ID) in a table
            with doc.create(Subsection('Parameters')):
                with doc.create(
                        LongTable('|c|c|c|',
                                  booktabs=True)) as data_table:
                    data_table.add_hline()
                    data_table.add_row(["Indexed by", "Description", "ID"])
                    data_table.add_hline()
                for parameter in decision_process_data.parameters:
                    param_shape = (
                        parameter.shape,
                    ) if isinstance(
                        parameter.shape,
                        IndexSet) else parameter.shape
                    index_seq_str = r"\rightarrow".join(
                        [idxset_id_mapping.get(p.id, p.id)
                         for p in param_shape]
                    )
                    if index_seq_str == "":
                        index_seq_str = r"\emptyset"
                    alias = cont_id_mapping.get(
                        parameter.id, parameter.id)
                    data_table.add_row([Math(data=index_seq_str,
                                             inline=True,
                                             escape=False),
                                        parameter.description,
                                        Math(data=alias,
                                             inline=True,
                                             escape=False)])
                    data_table.add_hline()
                data_table.add_hline()

            # Report the state variables informations
            # (shape, type, bounds, description, ID) in a table
            with doc.create(Subsection('State variables')):
                with doc.create(LongTable('|c|c|c|c|',
                                          booktabs=True)) as data_table:
                    data_table.add_hline()
                    data_table.add_row(["Indexed by",
                                        "Bounds",
                                        "Description",
                                        "ID"])
                    data_table.add_hline()
                for state_variable in decision_process_data.state_variables:
                    state_shape = (
                        state_variable.shape,
                    ) if isinstance(
                        state_variable.shape,
                        IndexSet) else state_variable.shape
                    index_seq_str = r"\rightarrow".join(
                        [idxset_id_mapping.get(p.id, p.id)
                         for p in state_shape]
                    )
                    if index_seq_str == "":
                        index_seq_str = r"\emptyset"
                    alias = cont_id_mapping.get(
                        state_variable.id,
                        state_variable.id)
                    data_table.add_row(
                        [
                            Math(data=index_seq_str,
                                 inline=True,
                                 escape=False),
                            Math(
                                data=support_to_str(
                                    state_variable.v_type,
                                    state_variable.support),
                                inline=True,
                                escape=False),
                            state_variable.description,
                            Math(
                                data=alias,
                                inline=True,
                                escape=False)])
                    data_table.add_hline()
                data_table.add_hline()
            # Report the action variables informations
            # (shape, type, bounds, description, ID) in a table
            with doc.create(Subsection('Action variables')):
                with doc.create(LongTable('|c|c|c|c|',
                                          booktabs=True)) as data_table:
                    data_table.add_hline()
                    data_table.add_row(["Indexed by",
                                        "Bounds",
                                        "Description",
                                        "Alias"])
                    data_table.add_hline()
                for action_variable in decision_process_data.action_variables:
                    action_shape = (
                        action_variable.shape,
                    ) if isinstance(
                        action_variable.shape,
                        IndexSet) else action_variable.shape
                    index_seq_str = r"\rightarrow".join(
                        [idxset_id_mapping.get(p.id, p.id)
                         for p in action_shape]
                    )
                    if index_seq_str == "":
                        index_seq_str = r"\emptyset"
                    alias = cont_id_mapping.get(
                        action_variable.id,
                        action_variable.id)
                    data_table.add_row(
                        [
                            Math(data=index_seq_str,
                                 inline=True,
                                 escape=False),
                            Math(
                                data=support_to_str(
                                    action_variable.v_type,
                                    action_variable.support),
                                inline=True,
                                escape=False),
                            action_variable.description,
                            Math(
                                data=alias,
                                inline=True,
                                escape=False)])
                    data_table.add_hline()
                data_table.add_hline()

            # Report the helper variables informations
            # (shape, type, bounds, description, ID) in a table
            with doc.create(Subsection('Helper variables')):
                with doc.create(LongTable('|c|c|p{4.3cm}|c|',
                                          booktabs=True)) as data_table:
                    data_table.add_hline()
                    data_table.add_row(["Indexed by",
                                        "Bounds",
                                        "Description",
                                        "ID"])
                    data_table.add_hline()
                for helper_variable in decision_process_data.helper_variables:
                    helper_shape = (
                        helper_variable.shape,
                    ) if isinstance(
                        helper_variable.shape,
                        IndexSet) else helper_variable.shape
                    index_seq_str = r"\rightarrow".join(
                        [idxset_id_mapping.get(p.id, p.id)
                         for p in helper_shape]
                    )
                    if index_seq_str == "":
                        index_seq_str = r"\emptyset"
                    alias = cont_id_mapping.get(
                        helper_variable.id,
                        helper_variable.id)
                    data_table.add_row(
                        [
                            Math(data=index_seq_str,
                                 inline=True,
                                 escape=False),
                            Math(
                                data=support_to_str(
                                    helper_variable.v_type,
                                    helper_variable.support),
                                inline=True,
                                escape=False),
                            helper_variable.description,
                            Math(
                                data=alias,
                                inline=True,
                                escape=False)])
                    data_table.add_hline()
                data_table.add_hline()
                doc.append(NewPage())

            # Report the cost function informations
            # (Description, numerical expression,
            #  horizon time coefficient, ID)
            with doc.create(Subsection('Cost functions')):
                doc.append(
                    Math(
                        data=[
                            r"given \; T > 0, \forall",
                            "0 < t < T:"],
                        escape=False,
                        inline=True))
                with doc.create(LongTable('|p{3.2cm}|c|c|c|',
                                          booktabs=True)) as data_table:
                    data_table.add_hline()
                    data_table.add_row(["Description",
                                        "Expression",
                                        r"Horizon coeff",
                                        "ID"])
                    data_table.add_hline()
                    for cost_function in decision_process_data.cost_functions:
                        alias = func_id_mapping.get(
                            cost_function.id, cost_function.id)
                        data_table.add_row(
                            [
                                cost_function.description,
                                Math(
                                    data=numerical_expression_to_str(
                                        cost_function.cost_expression,
                                        idx_set_mapping=idxset_id_mapping,
                                        cont_id_mapping=cont_id_mapping,
                                        sub_idxset_id_mapping=sub_idxset_id_mapping),
                                    inline=True,
                                    escape=False),
                                temporal_mask_to_str(
                                    cost_function.horizon_mask),
                                Math(
                                    data=alias,
                                    inline=True,
                                    escape=False)])
                        data_table.add_hline()
                    data_table.add_hline()

            # Objective function formulation
            with doc.create(Subsection('Objective function')):
                preamble = r"given \; T > 0, \; minimizes:"
                formula = r"\sum^T_{t = 1}{("
                formatted_cfs = [
                    func_id_mapping.get(
                        cf.id,
                        cf.id) +
                    "_{t, T}" for cf in decision_process_data.cost_functions]
                formula += "+".join(formatted_cfs)
                formula += ")}"
                doc.append(Math(data=preamble, escape=False))
                doc.append(Math(data=formula, escape=False))

            doc.append(NewPage())

            # Dynamics set enumeration with description
            # and free index sets
            with doc.create(Subsection('Dynamics')):
                doc.append(
                    Math(
                        data=[
                            r"\forall",
                            "t > 0:"],
                        escape=False,
                        inline=True))
                with doc.create(Alignat(numbering=True, escape=False)) as agn:
                    i = 0
                    for dynamics in decision_process_data.dynamics_functions:
                        temp_sub_idxset_id_mapping = dict(sub_idxset_id_mapping)
                        idxset_forall_id_mapping = {
                            i.id: temp_sub_idxset_id_mapping.get(
                                      i.id, i.id[0]
                                  ).lower()
                            for i in dynamics.state_var.idx_seq
                        }
                        temp_sub_idxset_id_mapping.update(idxset_forall_id_mapping)
                        lhs = numerical_expression_to_str(
                            dynamics.state_var,
                            tdiff=1,
                            idx_set_mapping=idxset_id_mapping,
                            cont_id_mapping=cont_id_mapping,
                            sub_idxset_id_mapping=temp_sub_idxset_id_mapping)
                        rhs = numerical_expression_to_str(
                            dynamics.state_var_update,
                            idx_set_mapping=idxset_id_mapping,
                            cont_id_mapping=cont_id_mapping,
                            sub_idxset_id_mapping=temp_sub_idxset_id_mapping)
                        formula_str = r"\shortintertext{" + \
                            dynamics.description + r":} \nonumber \\"
                        if idxset_forall_id_mapping != dict():
                            formula_str += r'&\forall{'
                            for (idset,
                                 idsub) in idxset_forall_id_mapping.items():
                                formula_str += r'' + idsub + r' \in ' + \
                                    idxset_id_mapping.get(
                                        idset, idset
                                    ) + r", \;"
                            formula_str += r"}\nonumber \\"

                        formula_str += (lhs + " &= " + rhs)

                        if i < len(
                                decision_process_data.dynamics_functions) - 1:
                            formula_str += r'\\[10pt] \hline \nonumber \\[1pt]'
                        agn.append(formula_str)
                        i += 1
            doc.append(NewPage())

            # Constraint set enumeration with description
            # and free index sets
            with doc.create(Subsection('Constraints')):
                doc.append(
                    Math(
                        data=[
                            r"\forall",
                            "t > 0:"],
                        escape=False,
                        inline=True))
                with doc.create(Alignat(numbering=True, escape=False)) as agn:
                    i = 0
                    for constraint in\
                            decision_process_data.constraint_functions:
                        temp_sub_idxset_id_mapping = dict(sub_idxset_id_mapping)
                        free_idx_sets_lhs = free_index_sets(
                            constraint.ineq.expr_1)
                        free_idx_sets_rhs = free_index_sets(
                            constraint.ineq.expr_2)
                        free_idx_sets = free_idx_sets_lhs.union(
                            free_idx_sets_rhs)
                        idxset_forall_id_mapping = {
                            i.id: temp_sub_idxset_id_mapping.get(
                                i.id, i.id[0]).lower() for i in free_idx_sets}
                        temp_sub_idxset_id_mapping.update(idxset_forall_id_mapping)
                        lhs = numerical_expression_to_str(
                            constraint.ineq.expr_1,
                            idx_set_mapping=idxset_id_mapping,
                            cont_id_mapping=cont_id_mapping,
                            sub_idxset_id_mapping=temp_sub_idxset_id_mapping)
                        rhs = numerical_expression_to_str(
                            constraint.ineq.expr_2,
                            idx_set_mapping=idxset_id_mapping,
                            cont_id_mapping=cont_id_mapping,
                            sub_idxset_id_mapping=temp_sub_idxset_id_mapping)
                        ineq_sign = comparison_to_str(constraint.ineq.ineq_op)
                        if len(get_variables(
                                    constraint.ineq.expr_2)) + \
                                len(get_parameters(
                                        constraint.ineq.expr_2)) > 3:
                            line_break = r"\nonumber\\ &"
                        else:
                            line_break = ""
                        formula_str = r"\shortintertext{" + \
                            constraint.description + r":} \nonumber \\"
                        if idxset_forall_id_mapping != dict():
                            formula_str += r'&\forall{'
                            for (idset,
                                 idsub) in idxset_forall_id_mapping.items():
                                formula_str += r'' + idsub + r' \in ' + \
                                    idxset_id_mapping.get(idset,
                                                          idset) + r", \;"
                            formula_str += r"}\nonumber \\"
                        formula_str += (lhs + " &" +
                                        ineq_sign + line_break + rhs)
                        if i < len(
                                decision_process_data.constraint_functions
                                ) - 1:
                            formula_str +=\
                                r'\\[10pt] \hline  \nonumber \\[1pt]'
                        agn.append(formula_str)
                        i += 1

    # Generate the pdf (and the tex file is requested)
    doc.generate_pdf(path + decision_process_data.id.lower().replace(" ", "_"))
    if get_tex:
        doc.generate_tex(
            path +
            decision_process_data.id.lower().replace(
                " ",
                "_"))
