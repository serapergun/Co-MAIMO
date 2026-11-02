"""Measure B4-S4 feature sensitivity on frozen real-SUMO decision states.

Offline diagnostic only: do not interpret perturbed logits as new SUMO outcomes.
"""
import json
from pathlib import Path
import numpy as np
from scripts.audit_b4_s4_ablation import ROOT, OUT, SEEDS, probabilities

SCALES=(0.0,0.5,1.0,2.0,4.0,8.0)
FEATURES={"density":1,"soc":2}

def main():
    with np.load(ROOT/"checkpoints"/"b4_s4_final.npz",allow_pickle=False) as ckpt:
        weights=np.asarray(ckpt["weights"],dtype=float)
    OUT.mkdir(parents=True,exist_ok=True)
    summary=[]
    for feature,index in FEATURES.items():
        for scale in SCALES:
            w=weights.copy()
            w[index]*=scale
            total=0
            multi=0
            tv=[]
            argmax_changes=0
            for seed in SEEDS:
                path=ROOT/"heldout"/f"seed_{seed}"/"dispatch_decisions.json"
                for d in json.loads(path.read_text()):
                    mask=np.asarray(d["feasible_mask"],dtype=bool)
                    if mask.sum()<2:continue
                    multi+=1
                    full=probabilities(d["candidate_features"],mask,weights,"full")
                    variant=probabilities(d["candidate_features"],mask,w,"full")
                    tv.append(float(.5*np.abs(full-variant).sum()))
                    argmax_changes+=int(np.argmax(full)!=np.argmax(variant))
                    total+=1
            summary.append({"feature":feature,"weight_scale":scale,
                            "multi_candidate_decisions":multi,
                            "argmax_changed":argmax_changes,
                            "mean_probability_tv":float(np.mean(tv)),
                            "max_probability_tv":float(np.max(tv))})
    output={"design":"frozen real SUMO states; vary one feature weight at a time",
            "warning":"Diagnostic only: no new training, no SUMO rerun, no causal outcome estimate",
            "baseline_weights":weights.tolist(),"sensitivity":summary}
    (OUT/"weight_sensitivity.json").write_text(json.dumps(output,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=="__main__":main()
