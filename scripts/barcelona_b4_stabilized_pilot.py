"""Barcelona B4-S shared-score PPO: independent train and heldout SUMO evaluation.

B4-S is a new model, NOT the frozen B4 checkpoint. Never updates on heldout.
"""
from pathlib import Path
import json
import numpy as np
import sumolib
from scripts.barcelona_b4_mappo import NET,N_TAXIS,N_REQUESTS,POLICY_SEED,write_scenario,validate_episode
from src.data.barcelona_crn import safe_od_pool
from src.evaluation.phase3a_crn import DEFAULT_PHASE3A_SEEDS
from src.marl.shared_score_mappo import SharedScoreMAPPO
from src.marl.traci_adapter import MAPPOTraciDispatcher
from src.marl.stabilized_training import train_stabilized_episode
from src.simulation.traci_runner import run
from src.marl.checkpoint import checkpoint_sha256

ROOT=Path("artifacts/barcelona_b4_stabilized")
CKPT=ROOT/"checkpoints"/"b4_stabilized_final.npz"

def main():
    if not NET.exists():raise SystemExit("Validated Barcelona network missing")
    pairs=safe_od_pool(sumolib.net.readNet(str(NET)))
    if len(pairs)<12:raise ValueError("Insufficient Barcelona OD pairs")
    plan=DEFAULT_PHASE3A_SEEDS.validate()
    assert not (set(plan.training)&set(plan.heldout)), "training/heldout seed leakage"
    policy=SharedScoreMAPPO(N_TAXIS,state_dim=3,seed=POLICY_SEED)
    policy.actor_w[0]=-2.0  # Explicit nearest-access prior; trained, not frozen\n    history=[]
    history=[]
    for i,seed in enumerate(plan.training):
        cfg=write_scenario(ROOT/"scenarios"/f"train_{seed}",pairs,seed)
        out=ROOT/"train"/f"seed_{seed}"
        dispatcher=MAPPOTraciDispatcher(policy,seed=POLICY_SEED+i+1)
        outputs=run(cfg,out,dispatcher,max_time_s=18000)
        validate_episode(out)
        update=train_stabilized_episode(dispatcher,outputs["request_qc"],policy)
        history.append({"demand_seed":int(seed),**update,"actor_coefficient":float(policy.actor_w[0])})
    CKPT.parent.mkdir(parents=True,exist_ok=True)
    np.savez(CKPT,actor_w=policy.actor_w,critic_w=policy.critic_w,
             action_dim=policy.action_dim,state_dim=policy.state_dim,
             obs_dim=policy.obs_dim,architecture="shared_score_ppo_v1")
    # Independent frozen policy restored from the saved checkpoint.
    frozen=SharedScoreMAPPO(N_TAXIS,state_dim=3,seed=0)
    with np.load(CKPT,allow_pickle=False) as state:
        if str(state["architecture"])!="shared_score_ppo_v1":raise RuntimeError("checkpoint architecture mismatch")
        frozen.actor_w=np.asarray(state["actor_w"],dtype=float).copy()
        frozen.critic_w=np.asarray(state["critic_w"],dtype=float).copy()
    heldout=[]
    for i,seed in enumerate(plan.heldout):
        cfg=write_scenario(ROOT/"scenarios"/f"heldout_{seed}",pairs,seed)
        out=ROOT/"heldout"/f"seed_{seed}"
        dispatcher=MAPPOTraciDispatcher(frozen,seed=POLICY_SEED+10000+i)
        run(cfg,out,dispatcher,max_time_s=18000)
        metrics=validate_episode(out)
        metrics["demand_seed"]=int(seed)
        heldout.append(metrics)
        (out/"validated_metrics.json").write_text(json.dumps(metrics,indent=2))
        (out/"dispatch_decisions.json").write_text(json.dumps(dispatcher.decisions,indent=2))
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/"training_history.json").write_text(json.dumps(history,indent=2))
    (ROOT/"heldout_metrics.json").write_text(json.dumps(heldout,indent=2))
    (ROOT/"experiment_manifest.json").write_text(json.dumps({
        "benchmark":"B4-S2 stabilized shared-score PPO pilot",
        "checkpoint_sha256":checkpoint_sha256(CKPT),
        "architecture":"shared-score actor, nearest-time prior, return scaling and 4 PPO epochs",
        "training_demand_seeds":list(plan.training),
        "heldout_demand_seeds":list(plan.heldout),
        "training_updates_on_heldout":False,
        "taxi_count":N_TAXIS,"request_count":N_REQUESTS,
        "energy_model":"MMPEVEM",
        "warning":"Newly trained model; no performance improvement asserted until paired evaluation."
    },indent=2))
    print(json.dumps({"training":history,"heldout":heldout},indent=2))
if __name__=="__main__":main()
