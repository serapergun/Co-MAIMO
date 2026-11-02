"""Paired frozen-policy counterfactual score ablation on B4-S4 SUMO logs.

Same observed candidate sets, same learned weights, same uniforms; no SUMO
outcome or causal improvement claim is inferred from this offline diagnostic.
"""
import json
from pathlib import Path
import numpy as np

ROOT=Path("artifacts/barcelona_b4_s4")
OUT=Path("artifacts/barcelona_b4_s4_ablation")
SEEDS=(9101,9102,9103,9104,9105)
MODES=("full","without_density","without_soc","access_only")

def probabilities(features,mask,weights,mode):
    x=np.asarray(features,dtype=float).copy()
    m=np.asarray(mask,dtype=bool)
    if x.ndim!=2 or x.shape[1]!=5 or m.shape!=(len(x),) or not m.any():
        raise ValueError("invalid decision features")
    if mode in ("without_density","access_only"):
        x[:,1]=0
        x[:,3]=0
    if mode in ("without_soc","access_only"):
        x[:,2]=0
        x[:,4]=0
    if not np.isfinite(x[m]).all():raise ValueError("nonfinite features")
    logits=x[m]@np.asarray(weights,dtype=float)
    logits-=logits.max()
    e=np.exp(logits)
    p=np.zeros(len(m),dtype=float)
    p[m]=e/e.sum()
    return p

def main():
    with np.load(ROOT/"checkpoints"/"b4_s4_final.npz",allow_pickle=False) as state:
        weights=state["weights"].copy()
    OUT.mkdir(parents=True,exist_ok=True)
    results=[]
    for seed in SEEDS:
        decisions=json.loads((ROOT/"heldout"/f"seed_{seed}"/"dispatch_decisions.json").read_text())
        rng=np.random.default_rng(seed+750000)
        changes={mode:0 for mode in MODES[1:]}
        probability_shift={mode:[] for mode in MODES[1:]}
        for d in decisions:
            x=d["candidate_features"]
            mask=d["feasible_mask"]
            # Common random number for each decision and every ablation.
            u=float(rng.random())
            selections={}
            scores={}
            for mode in MODES:
                p=probabilities(x,mask,weights,mode)
                scores[mode]=p
                selections[mode]=int(np.searchsorted(np.cumsum(p),u,side="right"))
            for mode in MODES[1:]:
                changes[mode]+=int(selections[mode]!=selections["full"])
                probability_shift[mode].append(float(.5*np.abs(scores[mode]-scores["full"]).sum()))
        results.append({"seed":seed,"decisions":len(decisions),
                        "changed_actions_common_random":changes,
                        "mean_total_variation_probability":{
                            mode:float(np.mean(probability_shift[mode])) for mode in MODES[1:]}})
    (OUT/"offline_ablation.json").write_text(json.dumps({
        "design":"frozen B4-S4 checkpoint, same SUMO candidate features and common uniform per decision",
        "warning":"Offline counterfactual choices; NOT rerun SUMO trajectories, rewards or performance.",
        "weights":weights.tolist(),"results":results},indent=2))
    print(json.dumps(results,indent=2))

if __name__=="__main__":main()
