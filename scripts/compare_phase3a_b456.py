"""Create paired B4-B5-B6 held-out comparison from validated CRN outputs."""
import argparse, csv, json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("--b4",default="artifacts/barcelona_b4_mappo/heldout_metrics.json")
p.add_argument("--b5",default="artifacts/barcelona_b5_dcf/heldout_metrics.json")
p.add_argument("--b6",default="artifacts/barcelona_b6_shapley/heldout_metrics.json")
p.add_argument("--out",default="artifacts/phase3a_b456")
p.add_argument("--b4-manifest",default="artifacts/barcelona_b4_mappo/experiment_manifest.json")
p.add_argument("--b5-manifest",default="artifacts/barcelona_b5_dcf/experiment_manifest.json")
p.add_argument("--b6-manifest",default="artifacts/barcelona_b6_shapley/experiment_manifest.json")
a=p.parse_args()

def load(path):
    rows=json.loads(Path(path).read_text())
    return {int(r["demand_seed"]):r for r in rows}

data={"B4":load(a.b4),"B5":load(a.b5),"B6":load(a.b6)}
manifests={
    "B4":json.loads(Path(a.b4_manifest).read_text()),
    "B5":json.loads(Path(a.b5_manifest).read_text()),
    "B6":json.loads(Path(a.b6_manifest).read_text()),
}
expected={9101,9102,9103,9104,9105}
for method in ("B4","B5","B6"):
    seeds=set(map(int,manifests[method].get("heldout_demand_seeds",[])))
    if seeds != expected:
        raise SystemExit(f"{method} manifest held-out seeds mismatch: {seeds}")
if manifests["B4"].get("training_updates_on_heldout") is not False:
    raise SystemExit("B4 manifest does not certify frozen held-out evaluation")
if manifests["B5"].get("policy_retrained") is not False:
    raise SystemExit("B5 manifest does not certify frozen B4 policy")
b5_hash=manifests["B5"].get("policy_sha256")
b6_hash=manifests["B6"].get("initial_policy_sha256")
if not b5_hash or not b6_hash or b5_hash != b6_hash:
    raise SystemExit("B5/B6 B4 checkpoint provenance mismatch")
for method,rows in data.items():
    if set(rows)!=expected:
        raise SystemExit(f"{method} held-out seeds mismatch: {set(rows)}")
    for seed,r in rows.items():
        if r["completed"]!=24 or r["unresolved"]!=0 or r["teleports"]!=0:
            raise SystemExit(f"{method} lifecycle QC failed for {seed}: {r}")

paired=[]
for seed in sorted(expected):
    row={"demand_seed":seed}
    for method in ("B4","B5","B6"):
        r=data[method][seed]
        for metric in ("mean_pickup_wait_s","empty_km","net_energy_kWh","kWh_per_vehicle_km"):
            row[f"{method}_{metric}"]=r[metric]
        row[f"{method}_accepted_coalitions"]=r.get("accepted_coalitions",0)
    for metric in ("mean_pickup_wait_s","empty_km","net_energy_kWh","kWh_per_vehicle_km"):
        row[f"B5_minus_B4_{metric}"]=row[f"B5_{metric}"]-row[f"B4_{metric}"]
        row[f"B6_minus_B5_{metric}"]=row[f"B6_{metric}"]-row[f"B5_{metric}"]
        row[f"B6_minus_B4_{metric}"]=row[f"B6_{metric}"]-row[f"B4_{metric}"]
    paired.append(row)

out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
(out/"paired_b456.json").write_text(json.dumps(paired,indent=2),encoding="utf-8")
with (out/"paired_b456.csv").open("w",newline="",encoding="utf-8") as fh:
    w=csv.DictWriter(fh,fieldnames=list(paired[0]))
    w.writeheader(); w.writerows(paired)

summary={}
for method in ("B4","B5","B6"):
    summary[method]={}
    for metric in ("mean_pickup_wait_s","empty_km","net_energy_kWh","kWh_per_vehicle_km"):
        vals=[data[method][s][metric] for s in sorted(expected)]
        summary[method][metric]={"mean":sum(vals)/len(vals),"min":min(vals),"max":max(vals)}
summary["design"]={
    "heldout_seeds":sorted(expected),
    "paired_crn":True,
    "n_heldout":len(expected),
    "statistical_claim":"descriptive paired multi-seed evaluation; inferential tests are not produced here",
    "provenance_verified":True,
    "shared_b4_checkpoint_sha256":b5_hash,
}
(out/"summary_b456.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(json.dumps(summary,indent=2))
