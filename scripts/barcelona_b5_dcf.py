"""Held-out B5 evaluation: frozen B4 MAPPO + DCF/CCE, no retraining."""
from pathlib import Path
import json
import sumolib

from src.data.barcelona_crn import safe_od_pool
from src.evaluation.phase3a_crn import DEFAULT_PHASE3A_SEEDS
from src.marl.checkpoint import load_checkpoint, load_checkpoint_metadata, checkpoint_sha256
from src.coalition.traci_dcf_adapter import DCFMAPPOTraciDispatcher
from src.simulation.traci_runner import run

# Reuse B4 scenario construction and physical validation verbatim so B4/B5
# differ only by the coalition mechanism.
from scripts.barcelona_b4_mappo import write_scenario, validate_episode, NET, N_TAXIS, N_REQUESTS


ROOT = Path("artifacts/barcelona_b5_dcf")
B4_CHECKPOINT = Path("artifacts/barcelona_b4_mappo/checkpoints/b4_mappo_final.npz")
POLICY_SEED = 4200


def main():
    if not NET.exists():
        raise SystemExit("Barcelona preflight network missing")
    if not B4_CHECKPOINT.exists():
        raise SystemExit("Frozen B4 checkpoint missing; B5 must reuse B4 policy")

    net = sumolib.net.readNet(str(NET))
    pairs = safe_od_pool(net)
    plan = DEFAULT_PHASE3A_SEEDS.validate()
    meta = load_checkpoint_metadata(B4_CHECKPOINT)
    if meta.get("benchmark") != "B4":
        raise SystemExit("B5 requires a checkpoint explicitly tagged benchmark=B4")
    if tuple(meta.get("training_demand_seeds", ())) != tuple(plan.training):
        raise SystemExit("B4 checkpoint training seeds do not match Phase3A CRN plan")
    if tuple(meta.get("heldout_demand_seeds", ())) != tuple(plan.heldout):
        raise SystemExit("B4 checkpoint held-out seeds do not match Phase3A CRN plan")
    policy = load_checkpoint(B4_CHECKPOINT)
    if policy.action_dim != N_TAXIS or policy.obs_dim != N_TAXIS or policy.state_dim != 3:
        raise SystemExit("B4 checkpoint dimensions incompatible with B5 protocol")
    checkpoint_hash = checkpoint_sha256(B4_CHECKPOINT)
    heldout = []

    for i, demand_seed in enumerate(plan.heldout):
        scenario = ROOT / "scenarios" / f"heldout_{demand_seed}"
        cfg = write_scenario(scenario, pairs, demand_seed)
        out = ROOT / "heldout" / f"seed_{demand_seed}"
        dispatcher = DCFMAPPOTraciDispatcher(
            policy,
            seed=POLICY_SEED + 10000 + i,
            coordination_cost_s=0.0,
        )
        run(cfg, out, dispatcher, max_time_s=18000)
        metrics = validate_episode(out)
        metrics["demand_seed"] = demand_seed
        metrics["coalition_events"] = len(dispatcher.coalition_events)
        metrics["accepted_coalitions"] = sum(
            1 for event in dispatcher.coalition_events if event["accepted"]
        )
        heldout.append(metrics)
        (out / "coalition_events.json").write_text(
            json.dumps(dispatcher.coalition_events, indent=2), encoding="utf-8"
        )
        (out / "validated_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )

    (ROOT / "heldout_metrics.json").write_text(
        json.dumps(heldout, indent=2), encoding="utf-8"
    )
    (ROOT / "experiment_manifest.json").write_text(json.dumps({
        "benchmark": "B5_MAPPO_DCF_CCE",
        "policy_source": str(B4_CHECKPOINT),
        "policy_sha256": checkpoint_hash,
        "policy_retrained": False,
        "heldout_demand_seeds": list(plan.heldout),
        "coordination_cost_s": 0.0,
        "taxi_count": N_TAXIS,
        "request_count": N_REQUESTS,
        "comparison": "paired against B4 on identical demand and policy-sampling seeds",
    }, indent=2), encoding="utf-8")
    print(json.dumps(heldout, indent=2))


if __name__ == "__main__":
    main()
