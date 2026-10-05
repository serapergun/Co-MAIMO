"""B6 training: MAPPO + DCF with accepted-event Shapley reward feedback."""
from pathlib import Path
import json
import numpy as np
import sumolib

from src.data.barcelona_crn import safe_od_pool
from src.evaluation.phase3a_crn import DEFAULT_PHASE3A_SEEDS
from src.marl.checkpoint import load_checkpoint, save_checkpoint
from src.marl.training import dispatch_reward, build_transitions
from src.marl.causal_shapley import causal_b6_rewards
from src.coalition.traci_dcf_adapter import DCFMAPPOTraciDispatcher
from src.simulation.traci_runner import run
from scripts.barcelona_b4_mappo import write_scenario, validate_episode, NET


ROOT=Path("artifacts/barcelona_b6_shapley")
B4_CHECKPOINT=Path("artifacts/barcelona_b4_mappo/checkpoints/b4_mappo_final.npz")
POLICY_SEED=4200
LAMBDA_PHI=0.1


def train_episode(policy, dispatcher, qc_path):
    q=json.loads(Path(qc_path).read_text())
    first=q["request_first_seen_s"]; assigned=q["request_assigned_s"]; pickup=q["request_pickup_s"]
    base=[]; states=[]
    for d in dispatcher.decisions:
        rid=str(d["reservation_id"])
        if rid not in first or rid not in assigned or rid not in pickup:
            raise RuntimeError(f"incomplete lifecycle for {rid}")
        queue=float(assigned[rid])-float(first[rid])
        access=float(pickup[rid])-float(assigned[rid])
        if queue < 0 or access < 0:
            raise RuntimeError(f"invalid lifecycle ordering for {rid}")
        base.append(dispatch_reward(access,queue))
        if "global_state" not in d:
            raise RuntimeError(f"missing decision-time critic state for {rid}")
        states.append(np.asarray(d["global_state"],dtype=float))
    shaped,credit_audit=causal_b6_rewards(
        dispatcher.decisions,base,dispatcher.coalition_events,
        lambda_phi=LAMBDA_PHI,decay=0.9,
    )
    # A DCF swap changes the reservation context after MAPPO sampling.
    # Its stored log-prob therefore belongs to the original reservation and
    # cannot be used as an on-policy PPO transition for the applied swap.
    keep=[i for i,d in enumerate(dispatcher.decisions) if not d.get("dcf_swap_applied",False)]
    if not keep:
        raise RuntimeError("B6 episode has no on-policy transitions after DCF filtering")
    train_decisions=[dispatcher.decisions[i] for i in keep]
    train_rewards=[shaped[i] for i in keep]
    train_states=[states[i] for i in keep]
    transitions=build_transitions(train_decisions,train_rewards,train_states,policy,gamma=0.99)
    update=policy.update(transitions)
    update["on_policy_transition_count"]=len(keep)
    update["excluded_swapped_transition_count"]=len(dispatcher.decisions)-len(keep)
    update["prior_credit_transition_count"]=sum(
        1 for i in keep if credit_audit[i]["has_prior_credit"]
    )
    return update, base, shaped, credit_audit


def main():
    if not B4_CHECKPOINT.exists():
        raise SystemExit("B4 checkpoint required before B6 training")
    net=sumolib.net.readNet(str(NET)); pairs=safe_od_pool(net)
    plan=DEFAULT_PHASE3A_SEEDS.validate()
    policy=load_checkpoint(B4_CHECKPOINT)
    history=[]

    for i,seed in enumerate(plan.training):
        cfg=write_scenario(ROOT/"scenarios"/f"train_{seed}",pairs,seed)
        out=ROOT/"train"/f"seed_{seed}"
        dispatcher=DCFMAPPOTraciDispatcher(policy,seed=POLICY_SEED+i+1,coordination_cost_s=0.0)
        outputs=run(cfg,out,dispatcher,max_time_s=18000)
        validate_episode(out)
        update,base,shaped,credit_audit=train_episode(policy,dispatcher,outputs["request_qc"])
        history.append({
            "demand_seed":seed,
            **update,
            "coalition_events":len(dispatcher.coalition_events),
            "accepted_coalitions":sum(e["accepted"] for e in dispatcher.coalition_events),
            "mean_base_reward":sum(base)/len(base),
            "mean_shaped_reward":sum(shaped)/len(shaped),
            "prior_credit_transition_count":update["prior_credit_transition_count"],
        })
        (out/"shapley_credit_audit.json").write_text(json.dumps(credit_audit,indent=2))

    checkpoint=ROOT/"checkpoints"/"b6_mappo_dcf_shapley_final.npz"
    save_checkpoint(policy,checkpoint,{
        "benchmark":"B6",
        "initial_policy":str(B4_CHECKPOINT),
        "lambda_phi":LAMBDA_PHI,
        "training_demand_seeds":list(plan.training),
    })

    frozen=load_checkpoint(checkpoint); heldout=[]
    for i,seed in enumerate(plan.heldout):
        cfg=write_scenario(ROOT/"scenarios"/f"heldout_{seed}",pairs,seed)
        out=ROOT/"heldout"/f"seed_{seed}"
        dispatcher=DCFMAPPOTraciDispatcher(frozen,seed=POLICY_SEED+10000+i,coordination_cost_s=0.0)
        run(cfg,out,dispatcher,max_time_s=18000)
        m=validate_episode(out)
        m.update({
            "demand_seed":seed,
            "coalition_events":len(dispatcher.coalition_events),
            "accepted_coalitions":sum(e["accepted"] for e in dispatcher.coalition_events),
        })
        heldout.append(m)
        (out/"coalition_events.json").write_text(json.dumps(dispatcher.coalition_events,indent=2))
        (out/"validated_metrics.json").write_text(json.dumps(m,indent=2))

    (ROOT/"training_history.json").write_text(json.dumps(history,indent=2))
    (ROOT/"heldout_metrics.json").write_text(json.dumps(heldout,indent=2))
    (ROOT/"experiment_manifest.json").write_text(json.dumps({
        "benchmark":"B6_MAPPO_DCF_SHAPLEY",
        "lambda_phi":LAMBDA_PHI,
        "training_demand_seeds":list(plan.training),
        "heldout_demand_seeds":list(plan.heldout),
        "heldout_policy_updates":False,
        "shapley_scope":"accepted operational DCF events only",
    },indent=2))
    print(json.dumps({"training":history,"heldout":heldout},indent=2))


if __name__=="__main__":
    main()
