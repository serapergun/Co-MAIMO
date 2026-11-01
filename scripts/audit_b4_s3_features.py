"""Audit observed B4-S3 dispatch data before designing a richer policy.

Reads validated SUMO artifacts only. Does not invent traffic, SoC or energy
features, and never modifies a trained checkpoint.
"""
import argparse
import json
from pathlib import Path
import numpy as np

def audit(root):
    root=Path(root)
    heldout=json.loads((root/"heldout_metrics.json").read_text())
    history=json.loads((root/"training_history.json").read_text())
    if len(heldout)!=5 or len(history)!=8:
        raise ValueError("Expected five heldout and eight training episodes")
    report={"model":"B4-S3","heldout_seeds":[],"training_episodes":len(history)}
    counts={"decision_count":0,"single_feasible":0,"multi_feasible":0,
            "nearest_selected":0,"non_nearest_selected":0,"missing_decision_logs":0}
    gaps=[]
    multi_gaps=[]
    per_seed=[]
    for row in heldout:
        seed=int(row["demand_seed"])
        if (row["completed"],row["unresolved"],row["teleports"])!=(24,0,0):
            raise ValueError(f"Invalid heldout QC for seed {seed}")
        p=root/"heldout"/f"seed_{seed}"/"dispatch_decisions.json"
        if not p.exists():
            counts["missing_decision_logs"]+=1
            continue
        decisions=json.loads(p.read_text())
        if len(decisions)!=24:
            raise ValueError(f"Unexpected dispatch count for seed {seed}")
        seed_non_nearest=0
        seed_multi=0
        for d in decisions:
            obs=np.asarray(d["observation"],dtype=float)
            mask=np.asarray(d["feasible_mask"],dtype=bool)
            action=int(d["action"])
            if obs.shape!=mask.shape or not mask.any() or not mask[action] or not np.all(np.isfinite(obs[mask])):
                raise ValueError(f"Invalid decision for seed {seed}")
            n=int(mask.sum())
            counts["decision_count"]+=1
            counts["single_feasible" if n==1 else "multi_feasible"]+=1
            if n>1:seed_multi+=1
            minimum=float(obs[mask].min())
            gap=float(obs[action]-minimum)
            if gap<=1e-10:counts["nearest_selected"]+=1
            else:
                counts["non_nearest_selected"]+=1
                seed_non_nearest+=1
            gaps.append(gap)
            if n>1:multi_gaps.append(gap)
        report["heldout_seeds"].append(seed)
        per_seed.append({"seed":seed,"multi_feasible":seed_multi,"non_nearest":seed_non_nearest})
    report["action_diagnostics"]=counts
    report["per_seed_choice_diagnostics"]=per_seed
    report["multi_feasible_non_nearest_rate"]=counts["non_nearest_selected"]/counts["multi_feasible"] if counts["multi_feasible"] else None
    report["multi_feasible_mean_normalized_access_regret"]=float(np.mean(multi_gaps)) if multi_gaps else None
    report["mean_normalized_access_regret"]=float(np.mean(gaps)) if gaps else None
    report["note"]="Regret is normalized access-time difference, not measured SUMO pickup wait or kWh."
    report["feature_availability"]={
        "normalized_access_time":"observed in dispatch_decisions.json",
        "feasibility_mask":"observed in dispatch_decisions.json",
        "global_idle_fraction":"observed in dispatch_decisions.json",
        "battery_soc":"NOT recorded in current decision observations",
        "local_traffic_density":"NOT recorded in current decision observations",
        "candidate_energy_cost":"NOT recorded in current decision observations"
    }
    return report

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="artifacts/barcelona_b4_s3")
    ap.add_argument("--out",default="artifacts/barcelona_b4_s3_audit/feature_audit.json")
    args=ap.parse_args()
    report=audit(args.root)
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
