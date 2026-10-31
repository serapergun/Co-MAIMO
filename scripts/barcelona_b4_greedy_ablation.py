"""B4-G deterministic nearest-feasible held-out ablation; frozen checkpoint."""
from pathlib import Path
import json
import numpy as np
import sumolib
from scripts.barcelona_b4_mappo import NET, N_TAXIS, write_scenario, validate_episode
from src.data.barcelona_crn import safe_od_pool
from src.evaluation.phase3a_crn import DEFAULT_PHASE3A_SEEDS
from src.marl.checkpoint import load_checkpoint, checkpoint_sha256
from src.marl.traci_adapter import MAPPOTraciDispatcher, _route_time
from src.simulation.traci_runner import run

ROOT=Path("artifacts/barcelona_b4_greedy")
CHECKPOINT=Path("artifacts/barcelona_b4_mappo/checkpoints/b4_mappo_final.npz")

class GreedyFrozenDispatcher(MAPPOTraciDispatcher):
    """Same sequential reservation processing, route-feasible taxi pool; argmin travel time."""
    def __call__(self,traci,reservations,fleet,now):
        fleet=tuple(sorted(fleet))
        if len(fleet)!=self.policy.action_dim:
            raise ValueError("Frozen policy/fleet size mismatch")
        idle=set(traci.vehicle.getTaxiFleet(0))
        used=set()
        assignments=[]
        for r in sorted(reservations,key=lambda x:(x.reservationTime,x.id)):
            candidates=[]
            for vid in fleet:
                if vid in idle and vid not in used:
                    t=_route_time(traci,vid,r)
                    if t is not None:
                        candidates.append((float(t),vid))
            if not candidates:
                continue
            t,vid=min(candidates,key=lambda pair:(pair[0],pair[1]))
            assignments.append((vid,[r.id]))
            used.add(vid)
            self.decisions.append({"time":float(now),"reservation_id":str(r.id),
                                   "vehicle_id":vid,"selected_access_time_s":t,
                                   "feasible_taxis":len(candidates)})
        return assignments

def main():
    if not NET.exists() or not CHECKPOINT.exists():
        raise SystemExit("Frozen Barcelona network and B4 checkpoint required")
    policy=load_checkpoint(CHECKPOINT)
    if policy.action_dim!=N_TAXIS:
        raise RuntimeError("Checkpoint fleet dimension mismatch")
    pairs=safe_od_pool(sumolib.net.readNet(str(NET)))
    heldout=[]
    for i,seed in enumerate(DEFAULT_PHASE3A_SEEDS.validate().heldout):
        cfg=write_scenario(ROOT/"scenarios"/f"heldout_{seed}",pairs,seed)
        out=ROOT/"heldout"/f"seed_{seed}"
        dispatcher=GreedyFrozenDispatcher(policy,seed=4200+10000+i)
        run(cfg,out,dispatcher,max_time_s=18000)
        m=validate_episode(out)
        if len(dispatcher.decisions)!=24:
            raise RuntimeError(f"Expected 24 decisions, got {len(dispatcher.decisions)}")
        m["demand_seed"]=int(seed)
        heldout.append(m)
        (out/"validated_metrics.json").write_text(json.dumps(m,indent=2))
        (out/"dispatch_decisions.json").write_text(json.dumps(dispatcher.decisions,indent=2))
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/"heldout_metrics.json").write_text(json.dumps(heldout,indent=2))
    (ROOT/"experiment_manifest.json").write_text(json.dumps({
        "benchmark":"B4-G deterministic nearest-feasible ablation",
        "b4_checkpoint_sha256":checkpoint_sha256(CHECKPOINT),
        "policy_retrained":False,
        "policy_logits_used":False,
        "heldout_seeds":list(DEFAULT_PHASE3A_SEEDS.validate().heldout),
        "note":"B4-G is a greedy dispatch ablation, not an improved learned MAPPO policy."
    },indent=2))
    print(json.dumps(heldout,indent=2))
if __name__=="__main__":
    main()
