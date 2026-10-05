"""Held-out B5 evaluation: frozen B4 MAPPO + DCF/CCE, no retraining."""
from pathlib import Path
import json
import sumolib

from src.data.barcelona_crn import safe_od_pool
from src.evaluation.phase3a_crn import DEFAULT_PHASE3A_SEEDS
from src.marl.checkpoint import load_checkpoint
from src.coalition.traci_dcf_adapter import DCFMAPPOTraciDispatcher
from src.simulation.traci_runner import run

# Reuse B4 scenario construction and physical validation verbatim so B4/B5
# differ only by the coalition mechanism.
from scripts.barcelona_b4_mappo import write_scenario, validate_episode, NET, N_TAXIS


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
    policy = load_checkpoint(B4_CHECKPOINT)
    plan = DEFAULT_PHASE3A_SEEDS.validate()
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
        "policy_retrained": False,
        "heldout_demand_seeds": list(plan.heldout),
        "coordination_cost_s": 0.0,
        "comparison": "paired against B4 on identical demand and policy-sampling seeds",
    }, indent=2), encoding="utf-8")
    print(json.dumps(heldout, indent=2))


if __name__ == "__main__":
    main()
