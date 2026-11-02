"""Audit candidate-level identifiability of B4-S4 features from real SUMO logs."""
import json
from pathlib import Path
import numpy as np
from scripts.audit_b4_s4_ablation import ROOT,OUT,SEEDS

def main():
    with np.load(ROOT/"checkpoints"/"b4_s4_final.npz",allow_pickle=False) as ck:
        weights=np.asarray(ck["weights"],dtype=float)
    spans={name:[] for name in ("access","density","soc")}
    spreads={name:[] for name in spans}
    argmax_nearest=0
    multi=0
    for seed in SEEDS:
        rows=json.loads((ROOT/"heldout"/f"seed_{seed}"/"dispatch_decisions.json").read_text())
        for d in rows:
            x=np.asarray(d["candidate_features"],dtype=float)
            mask=np.asarray(d["feasible_mask"],dtype=bool)
            if mask.sum()<2:continue
            v=x[mask]
            multi+=1
            for name,col in (("access",0),("density",1),("soc",2)):
                spans[name].append(float(np.ptp(v[:,col])))
                spreads[name].append(float(abs(weights[col])*np.ptp(v[:,col])))
            argmax_nearest+=int(np.argmax(v@weights)==np.argmin(v[:,0]))
    report={"multi_candidate_decisions":multi,"argmax_is_min_access":argmax_nearest,
            "weights":weights.tolist(),"features":{}}
    for name in spans:
        vals=np.asarray(spans[name])
        report["features"][name]={
            "nonzero_within_decision_spread":int((vals>1e-12).sum()),
            "median_within_decision_spread":float(np.median(vals)),
            "max_within_decision_spread":float(vals.max()),
            "median_weighted_logit_span":float(np.median(spreads[name])),
            "max_weighted_logit_span":float(max(spreads[name]))}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"feature_identifiability.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
