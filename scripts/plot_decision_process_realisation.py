from joblib import load
import plotly.graph_objects as go
from uliege.decision_process import\
    DecisionProcessRealisation, ContainerSequence
from typing import Dict, Union, List


def add_sequence(
    fig,
    c_sequence: ContainerSequence,
    cont_id_mapping: Dict[str, str] = dict(),
    idxset_id_mapping: Dict[str, str] = dict(),
    costfun_id_mapping: Dict[str, str] = dict(),
    is_cont: bool = True
):
    if is_cont:
        id_mapping = cont_id_mapping
    else:
        id_mapping = costfun_id_mapping
    for id, sequence in c_sequence.items():
        if not isinstance(sequence, dict):
            fig.add_trace(go.Scatter(
                y=sequence,
                name=r"$" +
                     id_mapping.get(id,
                                    id.replace("_",
                                               r"\_")).replace("$",
                                                               "") + "$"
            ))
        else:
            base_name = "$" +\
                        id_mapping.get(id,
                                       id.replace("_", r"\_")).replace("$",
                                                                       "")
            for idx_seq, values in sequence.items():
                name = base_name + "_{" + ",".join(
                    map(lambda x:
                        idxset_id_mapping.get(x.id,
                                              x.id.replace("_",
                                                           r"\_")).replace("$",
                                                                           ""),
                        idx_seq)) + "}"
                name += "$"
                fig.add_trace(go.Scatter(
                        y=values,
                        name=name
                    )
                )


def plot_decision_process_realisation(
    decision_process_realisation: DecisionProcessRealisation,
    cont_id_mapping: Dict[str, str] = dict(),
    idxset_id_mapping: Dict[str, str] = dict(),
    costfun_id_mapping: Dict[str, str] = dict(),
    log_scale: bool = False
):
    fig = go.Figure()
    state_sequence = decision_process_realisation.state_sequence
    action_sequence = decision_process_realisation.action_sequence
    helper_sequence = decision_process_realisation.helper_sequence
    parameter_sequence = decision_process_realisation.parameter_sequence
    cost_sequence = decision_process_realisation.cost_sequence
    lst_sequences = [
        (state_sequence, True),
        (action_sequence, True),
        (helper_sequence, True),
        (parameter_sequence, True),
        (cost_sequence, False)
    ]
    for sequence, is_cont in lst_sequences:
        add_sequence(fig,
                     sequence,
                     cont_id_mapping=cont_id_mapping,
                     idxset_id_mapping=idxset_id_mapping,
                     costfun_id_mapping=costfun_id_mapping,
                     is_cont=is_cont)
    if log_scale:
        fig.update_layout(yaxis_type="log")
    fig.show()


if __name__ == "__main__":
    dpr = load("sim_real.dump")
    plot_decision_process_realisation(
        dpr,
        log_scale=True
    )
