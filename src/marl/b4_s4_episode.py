"""Convert SUMO request QC and decision-time B4-S4 telemetry to PPO batches."""
import numpy as np
from src.marl.training import dispatch_reward,discounted_returns

def b4_s4_episode_batch(dispatcher,request_qc,policy,gamma=.99,return_scale=1000.):
    from pathlib import Path
    import json
    qc=json.loads(Path(request_qc).read_text()) if isinstance(request_qc,(str,Path)) else request_qc
    first_seen=qc.get("request_first_seen_s",{})
    assigned=qc.get("request_assigned_s",{})
    pickup=qc.get("request_pickup_s",{})
    decisions=dispatcher.decisions
    rewards=[]
    for d in decisions:
        rid=str(d["reservation_id"])
        if rid not in first_seen or rid not in assigned or rid not in pickup:
            raise ValueError("missing lifecycle QC for recorded dispatch decision")
        first=float(first_seen[rid])
        assigned_at=float(assigned[rid])
        pickup_at=float(pickup[rid])
        if assigned_at<first or pickup_at<assigned_at:
            raise ValueError("invalid lifecycle chronology")
        rewards.append(dispatch_reward(pickup_at-assigned_at,assigned_at-first))
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
