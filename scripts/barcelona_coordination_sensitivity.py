"""Coordination-overhead sensitivity for B5 on paired held-out CRN seeds.

Values are controlled protocol-overhead scenarios in seconds, not measured
communication latency. The purpose is to quantify how DCF acceptance and
operational outcomes change as coordination overhead increases.
"""
from pathlib import Path
import json
import sumolib

from src.data.barcelona_crn import safe_od_pool
from src.evaluation.phase3a_crn import DEFAULT_PHASE3A_SEEDS
from src.marl.checkpoint import load_checkpoint, load_checkpoint_metadata, checkpoint_sha256
from src.coalition.traci_dcf_adapter import DCFMAPPOTraciDispatcher
from src.simulation.traci_runner import run
from scripts.barcelona_b4_mappo import write_scenario, validate_episode, NET, N_TAXIS

ROOT=Path("artifacts/barcelona_coordination_sensitivity")
B4_CHECKPOINT=Path("artifacts/barcelona_b4_mappo/checkpoints/b4_mappo_final.npz")
POLICY_SEED=4200
OVERHEAD_S=(0.0,5.0,15.0,30.0)

def main():
    if not NET.exists() or not B4_CHECKPOINT.exists():
        raise SystemExit("Barcelona network and frozen B4 checkpoint are required")
    plan=DEFAULT_PHASE3A_SEEDS.validate()
    meta=load_checkpoint_metadata(B4_CHECKPOINT)
    if meta.get("benchmark")!="B4":
        raise SystemExit("sensitivity requires benchmark=B4 checkpoint")
    if tuple(meta.get("training_demand_seeds",())) != tuple(plan.training):
        raise SystemExit("B4 training seeds mismatch")
    if tuple(meta.get("heldout_demand_seeds",())) != tuple(plan.heldout):
        raise SystemExit("B4 held-out seeds mismatch")
    policy=load_checkpoint(B4_CHECKPOINT)
    if policy.action_dim != N_TAXIS or policy.obs_dim != N_TAXIS or policy.state_dim != 3:
        raise SystemExit("B4 checkpoint dimensions incompatible")
    b4_hash=checkpoint_sha256(B4_CHECKPOINT)
    pairs=safe_od_pool(sumolib.net.readNet(str(NET)))
    rows=[]
    for overhead in OVERHEAD_S:
        for i,seed in enumerate(plan.heldout):
            cfg=write_scenario(ROOT/"scenarios"/f"overhead_{int(overhead)}"/f"seed_{seed}",pairs,seed)
            out=ROOT/f"overhead_{int(overhead)}"/f"seed_{seed}"
            dispatcher=DCFMAPPOTraciDispatcher(
                policy,seed=POLICY_SEED+10000+i,coordination_cost_s=overhead
            )
            run(cfg,out,dispatcher,max_time_s=18000)
            m=validate_episode(out)
            events=dispatcher.coalition_events
            for event in events:
                threshold=float(event["effective_break_even_overhead_s"])
                # evaluate_merge requires strictly positive system gain and
                # non-negative member gains. Away from the numerical boundary,
                # the logged threshold must therefore agree with acceptance.
                if overhead < threshold-1e-9 and not event["accepted"]:
                    raise RuntimeError("DCF rejected below logged effective break-even overhead")
                if overhead > threshold+1e-9 and event["accepted"]:
                    raise RuntimeError("DCF accepted above logged effective break-even overhead")
            m.update({
                "demand_seed":seed,
                "coordination_overhead_s":overhead,
                "coalition_events":len(events),
                "accepted_coalitions":sum(bool(e["accepted"]) for e in events),
                "acceptance_rate":(
                    sum(bool(e["accepted"]) for e in events)/len(events) if events else 0.0
                ),
            })
            rows.append(m)
            (out/"coalition_events.json").write_text(json.dumps(events,indent=2))
            (out/"validated_metrics.json").write_text(json.dumps(m,indent=2))
    (ROOT/"sensitivity_rows.json").write_text(json.dumps(rows,indent=2))
    summary={}
    for overhead in OVERHEAD_S:
        subset=[r for r in rows if r["coordination_overhead_s"]==overhead]
        event_files=[
            ROOT/f"overhead_{int(overhead)}"/f"seed_{r['demand_seed']}"/"coalition_events.json"
            for r in subset
        ]
        events=[e for path in event_files for e in json.loads(path.read_text())]
        thresholds=[float(e["effective_break_even_overhead_s"]) for e in events]
        summary[str(int(overhead))]={
            "mean_acceptance_rate":sum(r["acceptance_rate"] for r in subset)/len(subset),
            "mean_accepted_coalitions":sum(r["accepted_coalitions"] for r in subset)/len(subset),
            "mean_pickup_wait_s":sum(r["mean_pickup_wait_s"] for r in subset)/len(subset),
            "mean_empty_km":sum(r["empty_km"] for r in subset)/len(subset),
            "mean_net_energy_kWh":sum(r["net_energy_kWh"] for r in subset)/len(subset),
            "proposal_count":len(events),
            "effective_break_even_overhead_s":{
                "mean":sum(thresholds)/len(thresholds) if thresholds else None,
                "min":min(thresholds) if thresholds else None,
                "max":max(thresholds) if thresholds else None,
            },
        }
    (ROOT/"sensitivity_summary.json").write_text(json.dumps(summary,indent=2))
    (ROOT/"experiment_manifest.json").write_text(json.dumps({
        "analysis":"DCF coordination-overhead sensitivity",
        "overhead_s":list(OVERHEAD_S),
        "overhead_interpretation":"controlled protocol-overhead scenarios; not measured network latency",
        "heldout_demand_seeds":list(plan.heldout),
        "policy_retrained":False,
        "policy_sha256":b4_hash,
        "paired_crn":True,
    },indent=2))
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
