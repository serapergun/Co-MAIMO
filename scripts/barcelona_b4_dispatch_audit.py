"""B4 held-out decision audit; no training or policy changes."""
from pathlib import Path
import csv
import json
import numpy as np
import sumolib
from scripts.barcelona_b4_mappo import write_scenario, validate_episode, NET, N_TAXIS
from src.data.barcelona_crn import safe_od_pool
from src.evaluation.phase3a_crn import DEFAULT_PHASE3A_SEEDS
from src.marl.checkpoint import load_checkpoint, checkpoint_sha256
from src.marl.traci_adapter import MAPPOTraciDispatcher, _route_time
from src.simulation.traci_runner import run

ROOT=Path("artifacts/b4_dispatch_audit")
CKPT=Path("artifacts/barcelona_b4_mappo/checkpoints/b4_mappo_final.npz")

class AuditedDispatcher(MAPPOTraciDispatcher):
    def __init__(self,policy,seed):
        super().__init__(policy,seed)
        self.audit=[]
    def __call__(self,traci,reservations,fleet,now):
        assigned=super().__call__(traci,reservations,fleet,now)
        by_id={str(r.id):r for r in reservations}
        for d in self.decisions[len(self.audit):]:
            reservation=by_id[str(d["reservation_id"])]
            mask=d["feasible_mask"]
            times={}
            for i,vid in enumerate(sorted(fleet)):
                if mask[i]:
                    t=_route_time(traci,vid,reservation)
                    if t is not None:
                        times[vid]=float(t)
            if not times:
                raise RuntimeError("selected decision has no feasible taxi")
            selected=str(d["vehicle_id"])
            if selected not in times:
                raise RuntimeError("selected taxi missing from feasible route times")
            best=min(times.values())
            self.audit.append({
                "seed":self.seed,"time_s":d["time"],"reservation_id":d["reservation_id"],
                "selected_taxi":selected,"selected_access_s":times[selected],
                "best_feasible_access_s":best,
                "access_regret_s":times[selected]-best,
                "is_nearest":int(abs(times[selected]-best)<1e-6),
                "feasible_taxis":len(times),
            })
        return assigned

def main():
    if not NET.exists() or not CKPT.exists():
        raise SystemExit("Network and frozen B4 checkpoint are required")
    policy=load_checkpoint(CKPT)
    if policy.action_dim!=N_TAXIS:
        raise RuntimeError("Unexpected checkpoint dimensions")
    net=sumolib.net.readNet(str(NET))
    pairs=safe_od_pool(net)
    rows=[]
    for i,seed in enumerate(DEFAULT_PHASE3A_SEEDS.validate().heldout):
        scenario=write_scenario(ROOT/"scenarios"/f"seed_{seed}",pairs,seed)
        dispatcher=AuditedDispatcher(policy,4200+10000+i)
        dispatcher.seed=int(seed)
        out=ROOT/"heldout"/f"seed_{seed}"
        run(scenario,out,dispatcher,max_time_s=18000)
        validate_episode(out)
        if len(dispatcher.audit)!=24:
            raise RuntimeError(f"Expected 24 decisions, found {len(dispatcher.audit)}")
        rows.extend(dispatcher.audit)
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/"b4_decision_audit.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    summary={"checkpoint_sha256":checkpoint_sha256(CKPT),
             "decision_count":len(rows),
             "mean_access_regret_s":float(np.mean([r["access_regret_s"] for r in rows])),
             "nearest_selection_rate":float(np.mean([r["is_nearest"] for r in rows])),
             "per_seed":{}}
    for seed in sorted({r["seed"] for r in rows}):
        a=[r for r in rows if r["seed"]==seed]
        summary["per_seed"][str(seed)]={
            "count":len(a),
            "mean_access_regret_s":float(np.mean([r["access_regret_s"] for r in a])),
            "nearest_selection_rate":float(np.mean([r["is_nearest"] for r in a]))}
    (ROOT/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=="__main__":
    main()
