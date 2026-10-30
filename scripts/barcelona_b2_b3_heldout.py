"""Held-out CRN B2/B3 evaluations, matched to B1/B4/B5/B6."""
import json
from pathlib import Path
import sumolib
from src.baselines.b2_b3_dispatch import b2_static_zone_dispatch,b3_threshold_dispatch
from src.data.barcelona_crn import safe_od_pool
from src.evaluation.phase3a_crn import DEFAULT_PHASE3A_SEEDS
from src.simulation.traci_runner import run
from scripts.barcelona_b4_mappo import NET,N_TAXIS,N_REQUESTS,write_scenario,validate_episode

ROOT=Path("artifacts/barcelona_b2_b3_heldout")
METHODS={"B2_static_two_zone":b2_static_zone_dispatch,
         "B3_backlog_threshold_2":b3_threshold_dispatch}

def main():
    if not NET.exists():
        raise SystemExit("Barcelona network missing")
    pairs=safe_od_pool(sumolib.net.readNet(str(NET)))
    if len(pairs)<12:
        raise SystemExit("Insufficient safe OD pool")
    plan=DEFAULT_PHASE3A_SEEDS.validate()
    for method,dispatcher in METHODS.items():
        rows=[]
        for seed in plan.heldout:
            cfg=write_scenario(ROOT/"scenarios"/f"heldout_{seed}",pairs,seed)
            out=ROOT/method/"heldout"/f"seed_{seed}"
            run(cfg,out,dispatcher,max_time_s=18000)
            metrics=validate_episode(out)
            metrics["demand_seed"]=seed
            rows.append(metrics)
            (out/"validated_metrics.json").write_text(json.dumps(metrics,indent=2))
        root=ROOT/method
        (root/"heldout_metrics.json").write_text(json.dumps(rows,indent=2))
        (root/"experiment_manifest.json").write_text(json.dumps({
            "benchmark":method,
            "heldout_demand_seeds":list(plan.heldout),
            "taxi_count":N_TAXIS,"request_count":N_REQUESTS,
            "sumo_energy_model":"MMPEVEM",
            "scenario_builder":"scripts.barcelona_b4_mappo.write_scenario",
            "validation":"scripts.barcelona_b4_mappo.validate_episode",
            "policy_retrained":False,
            "method_specification":("fixed SHA256 origin-edge 2-zone allocation with cross-zone fallback"
               if method.startswith("B2") else "global nearest feasible pair first if pending requests >=2, otherwise FIFO B1")
        },indent=2))
        print(json.dumps({"method":method,"heldout":rows},indent=2))

if __name__=="__main__":
    main()
