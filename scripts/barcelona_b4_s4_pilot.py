"""Train B4-S4 on independent Barcelona SUMO seeds and evaluate frozen checkpoint."""
import json
from pathlib import Path
import numpy as np
import sumolib
from scripts.barcelona_b4_mappo import NET,N_TAXIS,N_REQUESTS,POLICY_SEED,write_scenario,validate_episode
from src.data.barcelona_crn import safe_od_pool
from src.evaluation.phase3a_crn import DEFAULT_PHASE3A_SEEDS
from src.marl.b4_s4_training import TrainableB4S4Actor
from src.marl.b4_s4_traci_adapter import B4S4Dispatcher
from src.marl.b4_s4_episode import b4_s4_episode_batch
from src.simulation.traci_runner import run

ROOT=Path("artifacts/barcelona_b4_s4")
CKPT=ROOT/"checkpoints"/"b4_s4_final.npz"

def main():
    if not NET.exists():raise FileNotFoundError(NET)
    pairs=safe_od_pool(sumolib.net.readNet(str(NET)))
    plan=DEFAULT_PHASE3A_SEEDS.validate()
    if set(plan.training)&set(plan.heldout):raise RuntimeError("training and heldout seed leakage")
    policy=TrainableB4S4Actor()
    history=[]
    for i,seed in enumerate(plan.training):
        cfg=write_scenario(ROOT/"scenarios"/f"train_{seed}",pairs,seed)
        out=ROOT/"train"/f"seed_{seed}"
        dispatcher=B4S4Dispatcher(policy,seed=POLICY_SEED+i+1)
        outputs=run(cfg,out,dispatcher,max_time_s=18000)
        validate_episode(out)
        batch=b4_s4_episode_batch(dispatcher,outputs["request_qc"],policy)
        if not batch:raise RuntimeError("no B4-S4 training transitions")
        updates=[]
        for _ in range(4):
            updates.append(policy.update(batch))
        history.append({"demand_seed":int(seed),"transitions":len(batch),
                        "actor_weights":policy.weights.tolist(),"last_update":updates[-1]})
    CKPT.parent.mkdir(parents=True,exist_ok=True)
    np.savez(CKPT,weights=policy.weights,critic_w=policy.critic_w,
             architecture="b4_s4_shared_telemetry_ppo_v1")
    frozen=TrainableB4S4Actor()
    with np.load(CKPT,allow_pickle=False) as state:
        if str(state["architecture"])!="b4_s4_shared_telemetry_ppo_v1":
            raise ValueError("checkpoint architecture mismatch")
        frozen.weights=np.asarray(state["weights"],dtype=float).copy()
        frozen.critic_w=np.asarray(state["critic_w"],dtype=float).copy()
    heldout=[]
    for i,seed in enumerate(plan.heldout):
        cfg=write_scenario(ROOT/"scenarios"/f"heldout_{seed}",pairs,seed)
        out=ROOT/"heldout"/f"seed_{seed}"
        dispatcher=B4S4Dispatcher(frozen,seed=POLICY_SEED+10000+i)
        run(cfg,out,dispatcher,max_time_s=18000)
        metrics=validate_episode(out)
        metrics["demand_seed"]=int(seed)
        heldout.append(metrics)
        (out/"dispatch_decisions.json").write_text(json.dumps(dispatcher.decisions,indent=2))
        (out/"validated_metrics.json").write_text(json.dumps(metrics,indent=2))
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/"training_history.json").write_text(json.dumps(history,indent=2))
    (ROOT/"heldout_metrics.json").write_text(json.dumps(heldout,indent=2))
    (ROOT/"experiment_manifest.json").write_text(json.dumps({
        "benchmark":"B4-S4 observed telemetry PPO",
        "training_seeds":list(plan.training),"heldout_seeds":list(plan.heldout),
        "heldout_updates":False,"taxi_count":N_TAXIS,"request_count":N_REQUESTS,
        "energy_model":"SUMO MMPEVEM","actor":"shared five-feature masked PPO",
        "warning":"No improvement claim without paired inference"},indent=2))
    print(json.dumps({"training":history,"heldout":heldout},indent=2))

if __name__=="__main__":main()
