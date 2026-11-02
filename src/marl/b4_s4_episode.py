"""Convert SUMO request QC and decision-time B4-S4 telemetry to PPO batches."""
import numpy as np
from src.marl.training import dispatch_reward,discounted_returns

def b4_s4_episode_batch(dispatcher,request_qc,policy,gamma=.99,return_scale=1000.):
    from pathlib import Path
    import json
    qc=json.loads(Path(request_qc).read_text()) if isinstance(request_qc,(str,Path)) else request_qc
    rows=qc.get("requests",qc) if isinstance(qc,dict) else qc
    if isinstance(rows,dict):rows=list(rows.values())
    by_id={str(row.get("reservation_id",row.get("request_id",row.get("id")))):row for row in rows}
    decisions=dispatcher.decisions
    rewards=[]
    for d in decisions:
        row=by_id.get(str(d["reservation_id"]))
        if row is None:raise ValueError("missing request QC for recorded dispatch decision")
        assigned=float(row["request_assigned_s"])
        first=float(row["request_first_seen_s"])
        if assigned<first:raise ValueError("invalid assignment chronology")
        rewards.append(dispatch_reward(float(d["selected_access_time_s"]),assigned-first))
    returns=discounted_returns(rewards,gamma)
    transitions=[]
    for d,ret in zip(decisions,returns):
        state=np.asarray(d["global_state"],dtype=float)
        scaled=float(ret)/return_scale
        transitions.append({
            "candidate_features":d["candidate_features"],
            "feasible_mask":d["feasible_mask"],
            "action":d["action"],"old_log_prob":d["old_log_prob"],
            "global_state":state,"return_":scaled,
            "advantage":scaled-policy.value(state),
        })
    return transitions
