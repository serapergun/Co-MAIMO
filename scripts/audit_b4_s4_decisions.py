"""Compare heldout B4-S3 and B4-S4 decision records without rerunning SUMO.

Reports paired dispatch changes, feature availability and score identifiability.
"""
import json
from pathlib import Path
import numpy as np

S3=Path("artifacts/barcelona_b4_s3/heldout")
S4=Path("artifacts/barcelona_b4_s4/heldout")
OUT=Path("artifacts/barcelona_b4_s4_audit")

def read(seed,root):
    p=root/f"seed_{seed}"/"dispatch_decisions.json"
    rows=json.loads(p.read_text())
    return {str(r["reservation_id"]):r for r in rows}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for seed in range(9101,9106):
        a,b=read(seed,S3),read(seed,S4)
        if a.keys()!=b.keys():
            raise ValueError(f"unpaired request ids for seed {seed}")
        changes=[]
        feasible_counts=[]
        density_available=0
        soc_available=0
        density_values=[]
        soc_values=[]
        for rid in sorted(a):
            old,new=a[rid],b[rid]
            mask=np.asarray(new["feasible_mask"],dtype=bool)
            feats=np.asarray(new["candidate_features"],dtype=float)
            if feats.shape!=(len(mask),5) or not np.isfinite(feats[mask]).all():
                raise ValueError("invalid candidate feature log")
            feasible_counts.append(int(mask.sum()))
            density_available+=int(feats[mask,3].sum())
            soc_available+=int(feats[mask,4].sum())
            density_values.extend(feats[mask,1][feats[mask,3]>0].tolist())
            soc_values.extend(feats[mask,2][feats[mask,4]>0].tolist())
            if old["vehicle_id"]!=new["vehicle_id"]:
                changes.append({"reservation_id":rid,"s3_taxi":old["vehicle_id"],
                                "s4_taxi":new["vehicle_id"]})
        summaries.append({
            "seed":seed,"paired_decisions":len(a),
            "changed_taxi_count":len(changes),
            "multi_feasible_decisions":sum(n>1 for n in feasible_counts),
            "feasible_candidate_count":sum(feasible_counts),
            "density_observed_candidates":density_available,
            "soc_observed_candidates":soc_available,
            "normalized_density_min":min(density_values,default=None),
            "normalized_density_max":max(density_values,default=None),
            "soc_min":min(soc_values,default=None),
            "soc_max":max(soc_values,default=None),
            "changed_assignments":changes,
        })
    (OUT/"decision_audit.json").write_text(json.dumps(summaries,indent=2))
    print(json.dumps(summaries,indent=2))

if __name__=="__main__":main()
