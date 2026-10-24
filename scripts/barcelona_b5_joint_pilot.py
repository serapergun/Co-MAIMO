"""B5 bounded joint-assignment SUMO pilot, using a frozen B4 actor (NOT trained B5)."""
import json
from pathlib import Path
from itertools import combinations, permutations
import numpy as np
from src.marl.checkpoint import load_checkpoint
from src.marl.traci_adapter import _route_time, reservation_features
from src.simulation.traci_runner import run
from scripts.barcelona_b4_mappo import write_scenario, validate_episode, NET
from src.data.barcelona_crn import safe_od_pool
import sumolib

class B5PilotDispatcher:
    def __init__(self, policy, seed=19101, max_pending=4, max_coalition=3):
        self.policy=policy
        self.rng=np.random.default_rng(seed)
        self.max_pending=max_pending
        self.max_coalition=max_coalition
        self.decisions=[]
        self.offset=0

    def __call__(self, traci, reservations, fleet, now):
        fleet=tuple(sorted(fleet))
        if len(fleet)!=self.policy.action_dim:
            raise ValueError("B4 checkpoint action dimension / fleet mismatch")
        idle=tuple(sorted(traci.vehicle.getTaxiFleet(0)))
        ordered=sorted(reservations,key=lambda r:(r.reservationTime,str(r.id)))
        if not ordered or not idle: return []
        start=self.offset % len(ordered)
        window=(ordered[start:]+ordered[:start])[:self.max_pending]
        self.offset=(start+len(window))%len(ordered)
        options=[]
        for r in window:
            obs,mask=reservation_features(traci,r,fleet)
            probs=self.policy.action_probs(obs)
            opts=[]
            for i,taxi in enumerate(fleet):
                if taxi not in idle or not mask[i]: continue
                t=_route_time(traci,taxi,r)
                if t is None:continue
                opts.append((taxi,float(t),float(probs[i])))
            options.append(opts)
        candidates=[((),0.0)]
        for k in range(1,min(len(window),len(idle),self.max_coalition)+1):
            for selected in combinations(range(len(window)),k):
                for taxi_ids in permutations(idle,k):
                    chosen=[];utility=0.0;ok=True
                    for idx,taxi in zip(selected,taxi_ids):
                        match=next(((t,p) for tid,t,p in options[idx] if tid==taxi),None)
                        if match is None:ok=False;break
                        travel,prob=match
                        utility+=1200.0-travel+100.0*np.log(max(prob,1e-300))
                        chosen.append((taxi,str(window[idx].id)))
                    if ok:candidates.append((tuple(chosen),utility))
        scores=np.array([v for _,v in candidates],dtype=float)
        scores-=scores.max()
        probs=np.exp(scores);probs/=probs.sum()
        selected=int(self.rng.choice(len(candidates),p=probs))
        pairs,score=candidates[selected]
        self.decisions.append({"time":float(now),"pending":len(ordered),"window":len(window),
                               "candidate_actions":len(candidates),"selected_pairs":pairs,
                               "joint_log_prob":float(np.log(max(probs[selected],1e-300)))})
        return [(taxi,[rid]) for taxi,rid in pairs]

def main():
    root=Path("artifacts/barcelona_b5_pilot")
    root.mkdir(parents=True,exist_ok=True)
    checkpoint=Path("artifacts/barcelona_b4_mappo/checkpoints/b4_mappo_final.npz")
    if not checkpoint.exists():raise SystemExit("Missing B4 checkpoint")
    if not NET.exists():raise SystemExit("Missing Barcelona SUMO network")
    policy=load_checkpoint(checkpoint)
    net=sumolib.net.readNet(str(NET))
    pairs=safe_od_pool(net)
    cfg=write_scenario(root/"scenarios"/"heldout_9101",pairs,9101)
    dispatcher=B5PilotDispatcher(policy)
    out=root/"heldout"/"seed_9101"
    try:
        run(cfg,out,dispatcher,max_time_s=18000)
        metrics=validate_episode(out)
        result={"status":"PASS","seed":9101,"metrics":metrics,"decisions":len(dispatcher.decisions),
                "method":"B4 actor + bounded joint DCF; no B5 PPO training or CCE"}
    except Exception as exc:
        result={"status":"FAIL","seed":9101,"error":repr(exc),"decisions":len(dispatcher.decisions)}
    (root/"pilot_result.json").write_text(json.dumps(result,indent=2))
    (root/"dispatch_decisions.json").write_text(json.dumps(dispatcher.decisions,indent=2))
    print(json.dumps(result,indent=2))
    if result["status"]!="PASS":raise SystemExit(1)

if __name__=="__main__":main()
