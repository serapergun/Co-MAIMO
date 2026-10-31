"""Paired B4/B4-G validation on five held-out SUMO seeds."""
from pathlib import Path
import csv,json,math
import numpy as np
from scipy.stats import t as student_t
ROOT=Path("artifacts/barcelona_b4_greedy")
METRICS=("mean_pickup_wait_s","empty_km","net_energy_kWh","kWh_per_vehicle_km")
SEEDS=(9101,9102,9103,9104,9105)
def load(path):
    data=json.loads(Path(path).read_text())
    rows={int(r["demand_seed"]):r for r in data}
    if set(rows)!=set(SEEDS):
        raise ValueError(f"Seed mismatch: {path}")
    for seed,r in rows.items():
        if (int(r["completed"]),int(r["unresolved"]),int(r["teleports"]))!=(24,0,0):
            raise ValueError(f"QC failed: {path}, {seed}")
    return rows
def main():
    b4=load("artifacts/barcelona_b4_mappo/heldout_metrics.json")
    greedy=load(str(ROOT/"heldout_metrics.json"))
    b1_path=Path("artifacts/barcelona_b1_heldout_crn/heldout_metrics.json")
    results=[]
    for metric in METRICS:
        x=np.array([b4[s][metric] for s in SEEDS],float)
        y=np.array([greedy[s][metric] for s in SEEDS],float)
        d=y-x
        n=len(d);mean=float(d.mean());sd=float(d.std(ddof=1));se=sd/math.sqrt(n)
        tcrit=float(student_t.ppf(.975,n-1))
        p=float(2*student_t.sf(abs(mean/se),n-1)) if se>0 else (0. if mean else 1.)
        results.append({"metric":metric,"b4_mean":float(x.mean()),"b4_g_mean":float(y.mean()),
                        "mean_difference_g_minus_b4":mean,
                        "percent_change":float(100*mean/x.mean()),
                        "ci95_lower":mean-tcrit*se,"ci95_upper":mean+tcrit*se,
                        "paired_t_p_unadjusted":p,
                        "improved_seed_count":int((d<0).sum()),
                        "worsened_seed_count":int((d>0).sum()),
                        "paired_seed_differences":dict(zip(map(str,SEEDS),map(float,d)))})
    # Holm correction across the four operational outcomes.
    order=sorted(range(len(results)),key=lambda i:results[i]["paired_t_p_unadjusted"])
    running=0.
    for rank,i in enumerate(order):
        running=max(running,(len(results)-rank)*results[i]["paired_t_p_unadjusted"])
        results[i]["holm_adjusted_p"]=min(1.,running)
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/"paired_inference.json").write_text(json.dumps({"n":5,"seeds":SEEDS,"results":results},indent=2))
    with (ROOT/"paired_inference.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=[k for k in results[0] if k!="paired_seed_differences"])
        w.writeheader();w.writerows({k:v for k,v in r.items() if k!="paired_seed_differences"} for r in results)
    print(json.dumps(results,indent=2))
if __name__=="__main__":main()
