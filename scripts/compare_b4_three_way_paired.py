"""Paired inference for B4 sampled, B4 greedy, and frozen B4 argmax."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import t as student_t

ROOT=Path("artifacts")
PATHS={"B4":"barcelona_b4_mappo/heldout_metrics.json","B4-G":"barcelona_b4_greedy/heldout_metrics.json","B4-Argmax":"barcelona_b4_argmax/heldout_metrics.json"}
METRICS=["mean_pickup_wait_s","empty_km","net_energy_kWh","kWh_per_vehicle_km"]
SEEDS=[9101,9102,9103,9104,9105]

def load(label):
    rows=json.loads((ROOT/PATHS[label]).read_text())
    data={int(row["demand_seed"]):row for row in rows}
    if sorted(data)!=SEEDS or len(rows)!=5: raise ValueError(f"{label}: heldout seeds mismatch")
    for seed,row in data.items():
        if (row["completed"],row["unresolved"],row["teleports"])!=(24,0,0):
            raise ValueError(f"{label}: failed QC for {seed}")
    return data

def compare(a,b,da,db):
    results=[]
    for metric in METRICS:
        x=np.array([da[s][metric] for s in SEEDS],float)
        y=np.array([db[s][metric] for s in SEEDS],float)
        delta=y-x;n=len(delta);se=float(delta.std(ddof=1)/np.sqrt(n))
        tstat=float(delta.mean()/se) if se>0 else float("inf")
        p=float(2*student_t.sf(abs(tstat),df=n-1)) if se>0 else (1.0 if abs(float(delta.mean()))<1e-12 else 0.0)
        margin=float(student_t.ppf(.975,n-1)*se)
        results.append({"comparison":f"{b} minus {a}","metric":metric,
          "a_mean":float(x.mean()),"b_mean":float(y.mean()),
          "difference":float(delta.mean()),"percent_change":float(100*delta.mean()/x.mean()),
          "ci95":[float(delta.mean()-margin),float(delta.mean()+margin)],
          "p_unadjusted":p,"seed_differences":dict(zip(map(str,SEEDS),map(float,delta)))})
    order=sorted(range(len(results)),key=lambda i:results[i]["p_unadjusted"])
    adjusted=[0.0]*len(order);running=0.0
    for rank,i in enumerate(order):
        running=max(running,(len(order)-rank)*results[i]["p_unadjusted"])
        adjusted[i]=min(1.0,running)
    for i,row in enumerate(results):row["p_holm"]=adjusted[i]
    return results

def main():
    data={label:load(label) for label in PATHS}
    out=[]
    for a,b in [("B4","B4-Argmax"),("B4-G","B4-Argmax"),("B4","B4-G")]:
        out.extend(compare(a,b,data[a],data[b]))
    folder=ROOT/"barcelona_b4_comparison";folder.mkdir(exist_ok=True)
    (folder/"paired_inference.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
if __name__=="__main__":main()
