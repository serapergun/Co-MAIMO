"""Frozen B4 policy structural diagnostics: no SUMO rerun or checkpoint changes."""
import json
from pathlib import Path
import numpy as np
from src.marl.checkpoint import load_checkpoint,checkpoint_sha256
from src.marl.mappo import LinearMAPPO

ROOT=Path("artifacts/barcelona_b4_policy_diagnostics")
CKPT=Path("artifacts/barcelona_b4_mappo/checkpoints/b4_mappo_final.npz")
SEEDS=list(range(8))
def main():
    p=load_checkpoint(CKPT)
    if (p.obs_dim,p.action_dim)!=(8,8):raise ValueError("Unexpected frozen checkpoint architecture")
    w=np.asarray(p.actor_w,dtype=float)
    rng=np.random.default_rng(20261031)
    n=10000
    regret=[];agree=[];entropy=[];argmax_agree=[]
    for _ in range(n):
        # Controlled route-time vectors; diagnostic only, not real SUMO observations.
        times=rng.uniform(30,900,size=8)
        mask=rng.random(8)>.25
        if not mask.any():mask[0]=True
        obs=times/times.max()
        probs=p.masked_action_probs(obs,mask)
        nearest=int(np.argmin(np.where(mask,times,np.inf)))
        chosen=int(np.argmax(probs))
        regret.append(float(times[chosen]-times[nearest]))
        argmax_agree.append(int(chosen==nearest))
        entropy.append(float(-(probs[probs>0]*np.log(probs[probs>0])).sum()))
    # Equal-time symmetry: a taxi-label-equivariant policy should assign equal
    # probabilities to all feasible taxis when all features are identical.
    equal=np.ones(8)
    allmask=np.ones(8,dtype=bool)
    equal_probs=p.masked_action_probs(equal,allmask)
    report={"checkpoint_sha256":checkpoint_sha256(CKPT),
      "actor_weight_shape":list(w.shape),"actor_weight_frobenius_norm":float(np.linalg.norm(w)),
      "actor_weight_row_norms":[float(v) for v in np.linalg.norm(w,axis=1)],
      "equal_time_action_probs":equal_probs.tolist(),
      "equal_time_max_min_gap":float(equal_probs.max()-equal_probs.min()),
      "controlled_test_count":n,
      "argmax_nearest_agreement_fraction":float(np.mean(argmax_agree)),
      "argmax_access_regret_mean_s":float(np.mean(regret)),
      "argmax_access_regret_median_s":float(np.median(regret)),
      "mean_masked_entropy_nats":float(np.mean(entropy)),
      "limitations":"Synthetic controlled diagnostic inputs only; not heldout traffic performance, not proof of causal effect. Frozen model unchanged."}
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/"policy_diagnostics.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
