"""Create paired B4-B5-B6 held-out comparison from validated CRN outputs."""
import argparse, csv, json, math, statistics
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
design_counts={
    method:(manifests[method].get("taxi_count"),manifests[method].get("request_count"))
    for method in ("B4","B5","B6")
}
if len(set(design_counts.values())) != 1 or None in design_counts["B4"]:
    raise SystemExit(f"B4/B5/B6 experiment dimension mismatch: {design_counts}")
taxi_count,request_count=design_counts["B4"]
taxi_count=int(taxi_count); request_count=int(request_count)
if taxi_count <= 0 or request_count <= 0:
    raise SystemExit("invalid experiment dimensions in manifests")
for method in ("B4","B5","B6"):
    seeds=set(map(int,manifests[method].get("heldout_demand_seeds",[])))
    if seeds != expected:
        raise SystemExit(f"{method} manifest held-out seeds mismatch: {seeds}")
if manifests["B4"].get("training_updates_on_heldout") is not False:
    raise SystemExit("B4 manifest does not certify frozen held-out evaluation")
if manifests["B5"].get("policy_retrained") is not False:
    raise SystemExit("B5 manifest does not certify frozen B4 policy")
if manifests["B6"].get("heldout_policy_updates") is not False:
    raise SystemExit("B6 manifest does not certify frozen held-out evaluation")
if "shapley_feedback_observed" not in manifests["B6"]:
    raise SystemExit("B6 manifest lacks Shapley feedback exposure evidence")
b4_hash=manifests["B4"].get("checkpoint_sha256")
b5_hash=manifests["B5"].get("policy_sha256")
b6_hash=manifests["B6"].get("initial_policy_sha256")
b6_final_hash=manifests["B6"].get("final_policy_sha256")
if not b4_hash or not b5_hash or not b6_hash or len({b4_hash,b5_hash,b6_hash}) != 1:
    raise SystemExit("B4/B5/B6 checkpoint provenance mismatch")
if not b6_final_hash:
    raise SystemExit("B6 manifest lacks final trained policy fingerprint")
for method,rows in data.items():
    if set(rows)!=expected:
        raise SystemExit(f"{method} held-out seeds mismatch: {set(rows)}")
    for seed,r in rows.items():
        if int(r["completed"])!=request_count or int(r["unresolved"])!=0 or int(r["teleports"])!=0:
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
# Paired descriptive effect reporting. Lower is better for all four
# operational metrics. With only five held-out CRN seeds, these are descriptive
# robustness statistics rather than inferential significance tests.
effects={}
for comparison,left,right in (
    ("B5_minus_B4","B4","B5"),
    ("B6_minus_B5","B5","B6"),
    ("B6_minus_B4","B4","B6"),
):
    effects[comparison]={}
    for metric in ("mean_pickup_wait_s","empty_km","net_energy_kWh","kWh_per_vehicle_km"):
        diffs=[float(data[right][s][metric])-float(data[left][s][metric]) for s in sorted(expected)]
        mean_diff=sum(diffs)/len(diffs)
        pct_changes=[]
        for s in sorted(expected):
            baseline=float(data[left][s][metric])
            current=float(data[right][s][metric])
            if baseline != 0.0:
                pct_changes.append(100.0*(current-baseline)/baseline)
        sd_diff=statistics.stdev(diffs) if len(diffs)>1 else 0.0
        dz=(mean_diff/sd_diff) if sd_diff>0 else None
        effects[comparison][metric]={
            "mean_difference":mean_diff,
            "median_difference":statistics.median(diffs),
            "min_difference":min(diffs),
            "max_difference":max(diffs),
            "improved_seed_count":sum(d<0 for d in diffs),
            "worsened_seed_count":sum(d>0 for d in diffs),
            "tied_seed_count":sum(d==0 for d in diffs),
            "paired_standardized_effect_dz":dz,
            "mean_paired_percent_change":(sum(pct_changes)/len(pct_changes) if pct_changes else None),
            "median_paired_percent_change":(statistics.median(pct_changes) if pct_changes else None),
            "percent_change_n":len(pct_changes),
            "direction":"lower_is_better",
        }
summary["paired_effects"]=effects
summary["design"]={
    "heldout_seeds":sorted(expected),
    "taxi_count":taxi_count,
    "request_count":request_count,
    "paired_crn":True,
    "n_heldout":len(expected),
    "statistical_claim":"descriptive paired multi-seed evaluation; inferential tests are not produced here",
    "provenance_verified":True,
    "shared_b4_checkpoint_sha256":b4_hash,
    "b6_final_policy_sha256":b6_final_hash,
    "b6_shapley_feedback_observed":bool(manifests["B6"]["shapley_feedback_observed"]),
    "b6_aggregate_shapley_credit_coverage":manifests["B6"].get("aggregate_shapley_credit_coverage"),
    "b6_total_nonzero_shaping_transition_count":manifests["B6"].get("total_nonzero_shaping_transition_count"),
}
(out/"summary_b456.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(json.dumps(summary,indent=2))
