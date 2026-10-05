"""B6 Shapley reward-strength ablation under paired CRN seeds.

Every lambda arm starts from the identical frozen B4 checkpoint and uses the
same training and held-out demand seeds. Lambda=0 is the DCF-without-Shapley-
reward control; nonzero arms isolate reward-shaping strength.
"""
from pathlib import Path
import json
import numpy as np
import sumolib

from src.data.barcelona_crn import safe_od_pool
from src.evaluation.phase3a_crn import DEFAULT_PHASE3A_SEEDS
from src.marl.checkpoint import load_checkpoint, load_checkpoint_metadata, checkpoint_sha256, save_checkpoint
from src.marl.training import dispatch_reward, build_transitions
from src.marl.causal_shapley import causal_b6_rewards
from src.coalition.traci_dcf_adapter import DCFMAPPOTraciDispatcher
from src.simulation.traci_runner import run
from scripts.barcelona_b4_mappo import write_scenario, validate_episode, NET

ROOT=Path("artifacts/barcelona_b6_lambda_sensitivity")
B4_CHECKPOINT=Path("artifacts/barcelona_b4_mappo/checkpoints/b4_mappo_final.npz")
POLICY_SEED=4200
LAMBDAS=(0.0,0.05,0.1,0.2)

def train_episode(policy,dispatcher,qc_path,lambda_phi):
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
        states.append(np.asarray(d["global_state"],dtype=float))
    shaped,audit=causal_b6_rewards(
        dispatcher.decisions,base,dispatcher.coalition_events,
        lambda_phi=lambda_phi,decay=0.9,
    )
    keep=[i for i,d in enumerate(dispatcher.decisions) if not d.get("dcf_swap_applied",False)]
    if not keep:
        raise RuntimeError("lambda ablation episode has no on-policy transitions")
    transitions=build_transitions(
        [dispatcher.decisions[i] for i in keep],
        [shaped[i] for i in keep],
        [states[i] for i in keep],policy,gamma=0.99,
    )
    update=policy.update(transitions)
    deltas=[float(audit[i]["shaped_reward"])-float(audit[i]["base_reward"]) for i in keep]
    return {
        **update,
        "on_policy_transition_count":len(keep),
        "excluded_swapped_transition_count":len(dispatcher.decisions)-len(keep),
        "prior_credit_transition_count":sum(audit[i]["has_prior_credit"] for i in keep),
        "nonzero_shaping_transition_count":sum(abs(d)>1e-12 for d in deltas),
        "mean_absolute_shaping_delta":sum(abs(d) for d in deltas)/len(deltas),
    }

def main():
    if not B4_CHECKPOINT.exists():
        raise SystemExit("B4 checkpoint required")
    plan=DEFAULT_PHASE3A_SEEDS.validate()
    meta=load_checkpoint_metadata(B4_CHECKPOINT)
    if meta.get("benchmark")!="B4":
        raise SystemExit("lambda sensitivity requires benchmark=B4 checkpoint")
    if tuple(meta.get("training_demand_seeds",())) != tuple(plan.training):
        raise SystemExit("B4 training seeds mismatch")
    if tuple(meta.get("heldout_demand_seeds",())) != tuple(plan.heldout):
        raise SystemExit("B4 held-out seeds mismatch")
    b4_hash=checkpoint_sha256(B4_CHECKPOINT)
    pairs=safe_od_pool(sumolib.net.readNet(str(NET)))
    arm_summaries=[]
    for lam in LAMBDAS:
        policy=load_checkpoint(B4_CHECKPOINT)
        arm=ROOT/f"lambda_{lam:g}"
        history=[]
        for i,seed in enumerate(plan.training):
            cfg=write_scenario(arm/"scenarios"/f"train_{seed}",pairs,seed)
            out=arm/"train"/f"seed_{seed}"
            dispatcher=DCFMAPPOTraciDispatcher(policy,seed=POLICY_SEED+i+1,coordination_cost_s=0.0)
            outputs=run(cfg,out,dispatcher,max_time_s=18000)
            validate_episode(out)
            u=train_episode(policy,dispatcher,outputs["request_qc"],lam)
            u.update({
                "demand_seed":seed,
                "coalition_events":len(dispatcher.coalition_events),
                "accepted_coalitions":sum(bool(e["accepted"]) for e in dispatcher.coalition_events),
            })
            history.append(u)
        checkpoint=arm/"checkpoints"/"policy_final.npz"
        save_checkpoint(policy,checkpoint,{
            "benchmark":"B6_LAMBDA_SENSITIVITY",
            "lambda_phi":lam,
            "initial_policy_sha256":b4_hash,
            "training_demand_seeds":list(plan.training),
            "heldout_demand_seeds":list(plan.heldout),
        })
        final_hash=checkpoint_sha256(checkpoint)
        frozen=load_checkpoint(checkpoint)
        heldout=[]
        for i,seed in enumerate(plan.heldout):
            cfg=write_scenario(arm/"scenarios"/f"heldout_{seed}",pairs,seed)
            out=arm/"heldout"/f"seed_{seed}"
            dispatcher=DCFMAPPOTraciDispatcher(frozen,seed=POLICY_SEED+10000+i,coordination_cost_s=0.0)
            run(cfg,out,dispatcher,max_time_s=18000)
            m=validate_episode(out)
            m.update({"demand_seed":seed,"lambda_phi":lam})
            heldout.append(m)
        total_on=sum(int(r["on_policy_transition_count"]) for r in history)
        total_credit=sum(int(r["prior_credit_transition_count"]) for r in history)
        total_nonzero=sum(int(r["nonzero_shaping_transition_count"]) for r in history)
        if lam == 0.0 and total_nonzero != 0:
            raise RuntimeError("lambda=0 control produced nonzero Shapley reward shaping")
        summary={
            "lambda_phi":lam,
            "initial_policy_sha256":b4_hash,
            "final_policy_sha256":final_hash,
            "training_seed_count":len(history),
            "heldout_seed_count":len(heldout),
            "aggregate_shapley_credit_coverage":total_credit/total_on if total_on else 0.0,
            "total_nonzero_shaping_transition_count":total_nonzero,
            "shapley_feedback_observed":bool(total_nonzero>0),
            "mean_pickup_wait_s":sum(r["mean_pickup_wait_s"] for r in heldout)/len(heldout),
            "mean_empty_km":sum(r["empty_km"] for r in heldout)/len(heldout),
            "mean_net_energy_kWh":sum(r["net_energy_kWh"] for r in heldout)/len(heldout),
            "mean_energy_intensity_kWh_per_vehicle_km":sum(r["kWh_per_vehicle_km"] for r in heldout)/len(heldout),
        }
        (arm/"training_history.json").write_text(json.dumps(history,indent=2))
        (arm/"heldout_metrics.json").write_text(json.dumps(heldout,indent=2))
        (arm/"summary.json").write_text(json.dumps(summary,indent=2))
        arm_summaries.append(summary)
    (ROOT/"lambda_sensitivity_summary.json").write_text(json.dumps(arm_summaries,indent=2))
    (ROOT/"experiment_manifest.json").write_text(json.dumps({
        "analysis":"B6 Shapley reward-strength ablation",
        "lambda_phi":list(LAMBDAS),
        "lambda_zero_interpretation":"DCF control with no Shapley reward shaping",
        "shared_initial_policy_sha256":b4_hash,
        "training_demand_seeds":list(plan.training),
        "heldout_demand_seeds":list(plan.heldout),
        "paired_crn":True,
        "heldout_policy_updates":False,
        "selection_claim":"sensitivity/ablation only; no lambda is declared optimal from this grid",
    },indent=2))
    print(json.dumps(arm_summaries,indent=2))

if __name__=="__main__":
    main()
