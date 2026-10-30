"""B1 sequential nearest-feasible baseline on the identical B4-B6 held-out CRN protocol.

Uses the exact B4 scenario builder, QC metrics, network, fleet, demand and
seeds. No policy training or test-time tuning.
"""
from pathlib import Path
import json
import sumolib

from src.baselines.b1_dispatch import dispatch
from src.data.barcelona_crn import safe_od_pool
from src.evaluation.phase3a_crn import DEFAULT_PHASE3A_SEEDS
from src.simulation.traci_runner import run
from scripts.barcelona_b4_mappo import NET, N_TAXIS, N_REQUESTS, write_scenario, validate_episode

ROOT = Path("artifacts/barcelona_b1_heldout")


def main():
    if not NET.exists():
        raise SystemExit("Barcelona preflight network missing")
    pairs = safe_od_pool(sumolib.net.readNet(str(NET)))
    if len(pairs) < 12:
        raise SystemExit("safe Barcelona OD pool too small")
    plan = DEFAULT_PHASE3A_SEEDS.validate()
    rows = []
    for seed in plan.heldout:
        cfg = write_scenario(ROOT / "scenarios" / f"heldout_{seed}", pairs, seed)
        out = ROOT / "heldout" / f"seed_{seed}"
        run(cfg, out, dispatch, max_time_s=18000)
        metrics = validate_episode(out)
        metrics["demand_seed"] = seed
        rows.append(metrics)
        (out / "validated_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "heldout_metrics.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    (ROOT / "experiment_manifest.json").write_text(
        json.dumps({
            "benchmark": "B1_sequential_nearest_feasible",
            "heldout_demand_seeds": list(plan.heldout),
            "taxi_count": N_TAXIS,
            "request_count": N_REQUESTS,
            "sumo_energy_model": "MMPEVEM",
            "policy_retrained": False,
            "scenario_builder": "scripts.barcelona_b4_mappo.write_scenario",
            "validation": "scripts.barcelona_b4_mappo.validate_episode",
            "comparison_design": "paired CRN; identical B4-B6 held-out demand",
        }, indent=2), encoding="utf-8"
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
