"""Frozen B4 policy argmax ablation with the original feasible-action mask."""
from pathlib import Path
import json
import numpy as np
import sumolib
from scripts.barcelona_b4_mappo import NET,N_TAXIS,write_scenario,validate_episode
from src.data.barcelona_crn import safe_od_pool
from src.evaluation.phase3a_crn import DEFAULT_PHASE3A_SEEDS
from src.marl.checkpoint import load_checkpoint,checkpoint_sha256
from src.marl.traci_adapter import MAPPOTraciDispatcher,reservation_features,_route_time
from src.simulation.traci_runner import run

ROOT=Path("artifacts/barcelona_b4_argmax")
CHECKPOINT=Path("artifacts/barcelona_b4_mappo/checkpoints/b4_mappo_final.npz")

class FrozenArgmaxDispatcher(MAPPOTraciDispatcher):
    def __call__(self,traci,reservations,fleet,now):
        fleet=tuple(sorted(fleet))
        if len(fleet)!=self.policy.action_dim: raise ValueError("fleet mismatch")
        idle=set(traci.vehicle.getTaxiFleet(0));used=set();assignments=[]
        for r in sorted(reservations,key=lambda x:(x.reservationTime,x.id)):
            obs,route_mask=reservation_features(traci,r,fleet)
            mask=np.asarray([bool(route_mask[i] and v in idle and v not in used) for i,v in enumerate(fleet)])
            if not mask.any(): continue
            probs=self.policy.masked_action_probs(obs,mask)
            action=int(np.argmax(probs))
            if not mask[action]: raise AssertionError("argmax selected masked action")
            vid=fleet[action]
            assignments.append((vid,[r.id]));used.add(vid)
            self.decisions.append({"time":float(now),"reservation_id":str(r.id),
              "vehicle_id":vid,"action":action,"selected_probability":float(probs[action]),
              "feasible_count":int(mask.sum()),
              "selected_access_time_s":float(_route_time(traci,vid,r))})
        return assignments

def main():
    if not NET.exists() or not CHECKPOINT.exists(): raise SystemExit("Missing validated frozen inputs")
    policy=load_checkpoint(CHECKPOINT)
    if policy.action_dim!=N_TAXIS: raise RuntimeError("Checkpoint action dimension mismatch")
    pairs=safe_od_pool(sumolib.net.readNet(str(NET)))
    rows=[]
    seeds=DEFAULT_PHASE3A_SEEDS.validate().heldout
    for seed in seeds:
        cfg=write_scenario(ROOT/"scenarios"/f"heldout_{seed}",pairs,seed)
        out=ROOT/"heldout"/f"seed_{seed}"
        dispatcher=FrozenArgmaxDispatcher(policy,seed=seed)
        run(cfg,out,dispatcher,max_time_s=18000)
        m=validate_episode(out)
        if len(dispatcher.decisions)!=24: raise RuntimeError("Expected 24 decisions")
        m["demand_seed"]=int(seed);rows.append(m)
        (out/"validated_metrics.json").write_text(json.dumps(m,indent=2))
        (out/"dispatch_decisions.json").write_text(json.dumps(dispatcher.decisions,indent=2))
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/"heldout_metrics.json").write_text(json.dumps(rows,indent=2))
    (ROOT/"experiment_manifest.json").write_text(json.dumps({
      "benchmark":"B4 frozen policy deterministic argmax (not retrained)",
      "checkpoint_sha256":checkpoint_sha256(CHECKPOINT),
      "heldout_demand_seeds":list(seeds),
      "taxi_count":N_TAXIS,"request_count":24,
      "policy_retrained":False,"selection":"argmax of masked B4 action probabilities"
    },indent=2))
    print(json.dumps(rows,indent=2))
if __name__=="__main__":main()
