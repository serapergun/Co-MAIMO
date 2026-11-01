"""Reproducible paired heldout inference for B4-S2 vs B3/B4/B4-G.

Holm adjustment controls family-wise error across all twelve comparisons.
No post-hoc claim of significance from unadjusted p-values.
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t
SEEDS=(9101,9102,9103,9104,9105)
METRICS=("mean_pickup_wait_s","empty_km","net_energy_kWh","kWh_per_vehicle_km")
PATHS={
"B3":"artifacts/barcelona_b3/heldout_metrics.json",
"B4":"artifacts/barcelona_b4_mappo/heldout_metrics.json",
"B4-G":"artifacts/barcelona_b4_greedy/heldout_metrics.json",
"B4-S2":"artifacts/barcelona_b4_stabilized/heldout_metrics.json",
}
def load(label):
    p=Path(PATHS[label])
    if not p.exists():raise FileNotFoundError(f"Missing required validated source: {p}")
    rows=json.loads(p.read_text())
    data={int(row["demand_seed"]):row for row in rows}
    if len(rows)!=len(SEEDS) or set(data)!=set(SEEDS):raise ValueError(f"{label}: seed mismatch")
    for seed,row in data.items():
        if (row["completed"],row["unresolved"],row["teleports"])!=(24,0,0):
            raise ValueError(f"{label}: QC failed seed {seed}")
        if any(not np.isfinite(float(row[m])) for m in METRICS):
            raise ValueError(f"{label}: nonfinite metric seed {seed}")
    return data
def compare(reference,candidate,ref,alt):
    out=[]
    for metric in METRICS:
        x=np.array([ref[s][metric] for s in SEEDS],dtype=float)
        y=np.array([alt[s][metric] for s in SEEDS],dtype=float)
        d=y-x;n=len(d);mean=float(d.mean())
        se=float(d.std(ddof=1)/np.sqrt(n))
        if se==0:p=1.0 if mean==0 else 0.0
        else:p=float(2*student_t.sf(abs(mean/se),df=n-1))
        margin=float(student_t.ppf(.975,n-1)*se)
        out.append({"comparison":f"{candidate} minus {reference}",
                    "metric":metric,"reference_mean":float(x.mean()),
                    "candidate_mean":float(y.mean()),"mean_difference":mean,
                    "percent_change":float(100*mean/x.mean()),
                    "ci95":[mean-margin,mean+margin],
                    "p_unadjusted":p,
                    "paired_differences":dict(zip(map(str,SEEDS),map(float,d)))})
    return out
def holm(rows):
    indices=sorted(range(len(rows)),key=lambda i:rows[i]["p_unadjusted"])
    running=0.
    for rank,i in enumerate(indices):
        running=max(running,(len(indices)-rank)*rows[i]["p_unadjusted"])
        rows[i]["p_holm_12_tests"]=min(1.,running)
    return rows
def main():
    data={name:load(name) for name in PATHS}
    results=[]
    for name in ("B3","B4","B4-G"):
        results.extend(compare(name,"B4-S2",data[name],data["B4-S2"]))
    holm(results)
    out=Path("artifacts/barcelona_b4_s2_inference")
    out.mkdir(parents=True,exist_ok=True)
    (out/"paired_inference.json").write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))
if __name__=="__main__":main()
