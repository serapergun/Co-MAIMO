"""B6 training: MAPPO + DCF with accepted-event Shapley reward feedback."""
from pathlib import Path
import json
import numpy as np
import sumolib

from src.data.barcelona_crn import safe_od_pool
from src.evaluation.phase3a_crn import DEFAULT_PHASE3A_SEEDS
from src.marl.checkpoint import load_checkpoint, save_checkpoint, load_checkpoint_metadata, checkpoint_sha256
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
    on_policy_deltas=[
        float(credit_audit[i]["shaped_reward"])-float(credit_audit[i]["base_reward"])
        for i in keep
    ]
    update["nonzero_shaping_transition_count"]=sum(abs(d)>1e-12 for d in on_policy_deltas)
    update["mean_absolute_shaping_delta"]=(
        sum(abs(d) for d in on_policy_deltas)/len(on_policy_deltas)
        if on_policy_deltas else 0.0
    )
    return update, base, shaped, credit_audit


def main():
    if not NET.exists():
        raise SystemExit("Barcelona preflight network missing")
    if not B4_CHECKPOINT.exists():
        raise SystemExit("B4 checkpoint required before B6 training")
    net=sumolib.net.readNet(str(NET)); pairs=safe_od_pool(net)
    plan=DEFAULT_PHASE3A_SEEDS.validate()
    meta=load_checkpoint_metadata(B4_CHECKPOINT)
    if meta.get("benchmark")!="B4":
        raise SystemExit("B6 requires a checkpoint explicitly tagged benchmark=B4")
    if tuple(meta.get("training_demand_seeds",())) != tuple(plan.training):
        raise SystemExit("B4 checkpoint training seeds do not match Phase3A CRN plan")
    if tuple(meta.get("heldout_demand_seeds",())) != tuple(plan.heldout):
        raise SystemExit("B4 checkpoint held-out seeds do not match Phase3A CRN plan")
    policy=load_checkpoint(B4_CHECKPOINT)
    if policy.action_dim != 8 or policy.obs_dim != 8 or policy.state_dim != 3:
        raise SystemExit("B4 checkpoint dimensions incompatible with B6 protocol")
    b4_checkpoint_hash=checkpoint_sha256(B4_CHECKPOINT)
    history=[]

    for i,seed in enumerate(plan.training):
        cfg=write_scenario(ROOT/"scenarios"/f"train_{seed}",pairs,seed)
        out=ROOT/"train"/f"seed_{seed}"
        dispatcher=DCFMAPPOTraciDispatcher(policy,seed=POLICY_SEED+i+1,coordination_cost_s=0.0)
        outputs=run(cfg,out,dispatcher,max_time_s=18000)
        validate_episode(out)
        update,base,shaped,credit_audit=train_episode(policy,dispatcher,outputs["request_qc"])
        on_policy_count=int(update["on_policy_transition_count"])
        credit_count=int(update["prior_credit_transition_count"])
        history.append({
            "demand_seed":seed,
            **update,
            "coalition_events":len(dispatcher.coalition_events),
            "accepted_coalitions":sum(e["accepted"] for e in dispatcher.coalition_events),
            "mean_base_reward":sum(base)/len(base),
            "mean_shaped_reward":sum(shaped)/len(shaped),
            "prior_credit_transition_count":credit_count,
            "shapley_credit_coverage":(credit_count/on_policy_count if on_policy_count else 0.0),
            "nonzero_shaping_transition_count":update["nonzero_shaping_transition_count"],
            "mean_absolute_shaping_delta":update["mean_absolute_shaping_delta"],
        })
        (out/"shapley_credit_audit.json").write_text(json.dumps(credit_audit,indent=2))

    checkpoint=ROOT/"checkpoints"/"b6_mappo_dcf_shapley_final.npz"
    save_checkpoint(policy,checkpoint,{
        "benchmark":"B6",
        "initial_policy":str(B4_CHECKPOINT),
        "initial_policy_sha256":b4_checkpoint_hash,
        "lambda_phi":LAMBDA_PHI,
        "training_demand_seeds":list(plan.training),
        "heldout_demand_seeds":list(plan.heldout),
    })
    b6_checkpoint_hash=checkpoint_sha256(checkpoint)
    frozen=load_checkpoint(checkpoint)
    if checkpoint_sha256(checkpoint) != b6_checkpoint_hash:
        raise RuntimeError("B6 checkpoint changed before held-out evaluation")
    heldout=[]
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

    expected_training=set(map(int,plan.training))
    observed_training={int(r["demand_seed"]) for r in history}
    if observed_training != expected_training:
        raise RuntimeError(f"incomplete B6 training seeds: {observed_training}")
    if any(int(r["on_policy_transition_count"]) <= 0 for r in history):
        raise RuntimeError("B6 training contains an episode with no on-policy transitions")
    total_on_policy=sum(int(r["on_policy_transition_count"]) for r in history)
    total_credit=sum(int(r["prior_credit_transition_count"]) for r in history)
    total_nonzero=sum(int(r["nonzero_shaping_transition_count"]) for r in history)
    if total_credit > total_on_policy or total_nonzero > total_on_policy:
        raise RuntimeError("invalid B6 Shapley exposure counts")
    training_summary={
        "training_seed_count":len(history),
        "total_on_policy_transition_count":total_on_policy,
        "total_excluded_swapped_transition_count":sum(
            int(r["excluded_swapped_transition_count"]) for r in history
        ),
        "total_coalition_events":sum(int(r["coalition_events"]) for r in history),
        "total_accepted_coalitions":sum(int(r["accepted_coalitions"]) for r in history),
        "total_prior_credit_transition_count":total_credit,
        "aggregate_shapley_credit_coverage":(
            total_credit/total_on_policy if total_on_policy else 0.0
        ),
        "total_nonzero_shaping_transition_count":total_nonzero,
        "shapley_feedback_observed":bool(total_nonzero > 0),
        "seeds_with_prior_credit":sum(
            int(r["prior_credit_transition_count"]) > 0 for r in history
        ),
        "seeds_with_nonzero_shaping":sum(
            int(r["nonzero_shaping_transition_count"]) > 0 for r in history
        ),
        "interpretation":(
            "Shapley-informed learning is empirically evidenced only when "
            "nonzero on-policy shaping transitions are observed."
        ),
    }
    (ROOT/"training_history.json").write_text(json.dumps(history,indent=2))
    (ROOT/"training_summary.json").write_text(json.dumps(training_summary,indent=2))
    (ROOT/"heldout_metrics.json").write_text(json.dumps(heldout,indent=2))
    (ROOT/"experiment_manifest.json").write_text(json.dumps({
        "benchmark":"B6_MAPPO_DCF_SHAPLEY",
        "initial_policy_sha256":b4_checkpoint_hash,
        "final_policy_sha256":b6_checkpoint_hash,
        "lambda_phi":LAMBDA_PHI,
        "training_demand_seeds":list(plan.training),
        "heldout_demand_seeds":list(plan.heldout),
        "heldout_policy_updates":False,
        "shapley_scope":"accepted operational DCF events only",
        "shapley_credit_timing":"strictly prior event within the same episode",
        "shapley_credit_memory":"reset for every CRN demand seed",
        "cross_seed_learning":"policy parameters only",
        "shapley_feedback_observed":training_summary["shapley_feedback_observed"],
        "aggregate_shapley_credit_coverage":training_summary["aggregate_shapley_credit_coverage"],
        "total_nonzero_shaping_transition_count":training_summary["total_nonzero_shaping_transition_count"],
    },indent=2))
    print(json.dumps({"training":history,"heldout":heldout},indent=2))


if __name__=="__main__":
    main()
