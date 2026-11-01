"""B4-S4 read-only telemetry availability pilot on real Barcelona SUMO.

Uses the frozen B4-S3 policy; no training, reward changes, or performance claims.
"""
import json
from pathlib import Path
import numpy as np
import sumolib
from scripts.barcelona_b4_mappo import NET,N_TAXIS,POLICY_SEED,write_scenario,validate_episode
from src.data.barcelona_crn import safe_od_pool
from src.marl.multifeature_mappo import MultiFeatureMAPPO
from src.marl.traci_adapter import MAPPOTraciDispatcher
from src.marl.telemetry_probes import candidate_telemetry
from src.simulation.traci_runner import run

ROOT=Path("artifacts/barcelona_b4_s4_telemetry")
SOURCE=Path("artifacts/barcelona_b4_s3/checkpoints/b4_s3_final.npz")
SEED=9101

class TelemetryDispatcher:
    def __init__(self,policy):
        self.base=MAPPOTraciDispatcher(policy,seed=POLICY_SEED+10000)
        self.probes=[]
    def __call__(self,traci,reservations,fleet,now):
        if reservations:
            idle=sorted(traci.vehicle.getTaxiFleet(0))
            for vehicle_id in idle:
                sample=candidate_telemetry(traci,vehicle_id)
                self.probes.append({"time_s":float(now),"vehicle_id":vehicle_id,
                    **{name:{"available":probe.available,"value":probe.value,
                              "reason":probe.reason} for name,probe in sample.items()}})
        return self.base(traci,reservations,fleet,now)

def main():
    if not NET.exists():raise FileNotFoundError(NET)
    if not SOURCE.exists():raise FileNotFoundError(SOURCE)
    pairs=safe_od_pool(sumolib.net.readNet(str(NET)))
    policy=MultiFeatureMAPPO(N_TAXIS,state_dim=3,seed=0)
    with np.load(SOURCE,allow_pickle=False) as ck:
        if len(ck["actor_w"])!=2:raise ValueError("Expected two-feature B4-S3 actor")
        policy.actor_w=np.asarray(ck["actor_w"],dtype=float).copy()
        policy.critic_w=np.asarray(ck["critic_w"],dtype=float).copy()
    dispatcher=TelemetryDispatcher(policy)
    cfg=write_scenario(ROOT/"scenarios"/f"seed_{SEED}",pairs,SEED)
    out=ROOT/"run"/f"seed_{SEED}"
    run(cfg,out,dispatcher,max_time_s=18000)
    metrics=validate_episode(out)
    counts={}
    for name in ("lane_density","battery_soc"):
        samples=[r[name] for r in dispatcher.probes]
        available=[r["value"] for r in samples if r["available"]]
        counts[name]={"total":len(samples),"available":len(available),
                      "missing":len(samples)-len(available),
                      "availability_rate":len(available)/len(samples) if samples else None,
                      "observed_min":min(available) if available else None,
                      "observed_max":max(available) if available else None}
    report={"demand_seed":SEED,"policy":"frozen_B4-S3","training":False,
            "telemetry":counts,"validated_episode":metrics}
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/"availability_report.json").write_text(json.dumps(report,indent=2))
    (ROOT/"probe_records.json").write_text(json.dumps(dispatcher.probes,indent=2))
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
