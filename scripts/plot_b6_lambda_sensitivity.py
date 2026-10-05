"""Publication artifacts for paired B6 Shapley lambda sensitivity."""
from pathlib import Path
import csv, json, statistics
import matplotlib.pyplot as plt

ROOT=Path("artifacts/barcelona_b6_lambda_sensitivity")
OUT=ROOT/"publication"
LAMBDAS=(0.0,0.05,0.1,0.2)
SEEDS=(9101,9102,9103,9104,9105)
METRICS=("mean_pickup_wait_s","empty_km","net_energy_kWh","kWh_per_vehicle_km")

def load_arm(lam):
    path=ROOT/f"lambda_{lam:g}"/"heldout_metrics.json"
    rows=json.loads(path.read_text())
    by={int(r["demand_seed"]):r for r in rows}
    if set(by)!=set(SEEDS):
        raise SystemExit(f"incomplete held-out CRN for lambda={lam}: {set(by)}")
    return by

def main():
    arms={lam:load_arm(lam) for lam in LAMBDAS}
    summaries=json.loads((ROOT/"lambda_sensitivity_summary.json").read_text())
    summary_by={float(r["lambda_phi"]):r for r in summaries}
    if set(summary_by)!=set(LAMBDAS):
        raise SystemExit("lambda summary grid mismatch")
    if int(summary_by[0.0]["total_nonzero_shaping_transition_count"]) != 0:
        raise SystemExit("lambda=0 control has nonzero shaping")

    table=[]
    paired=[]
    control=arms[0.0]
    for lam in LAMBDAS:
        row={
            "lambda_phi":lam,
            "shapley_feedback_observed":bool(summary_by[lam]["shapley_feedback_observed"]),
            "shapley_credit_coverage":float(summary_by[lam]["aggregate_shapley_credit_coverage"]),
            "nonzero_shaping_transitions":int(summary_by[lam]["total_nonzero_shaping_transition_count"]),
        }
        for metric in METRICS:
            vals=[float(arms[lam][s][metric]) for s in SEEDS]
            row[f"mean_{metric}"]=sum(vals)/len(vals)
            if lam != 0.0:
                diffs=[float(arms[lam][s][metric])-float(control[s][metric]) for s in SEEDS]
                row[f"mean_delta_vs_lambda0_{metric}"]=sum(diffs)/len(diffs)
                row[f"median_delta_vs_lambda0_{metric}"]=statistics.median(diffs)
                row[f"improved_seeds_vs_lambda0_{metric}"]=sum(d<0 for d in diffs)
                row[f"worsened_seeds_vs_lambda0_{metric}"]=sum(d>0 for d in diffs)
                row[f"tied_seeds_vs_lambda0_{metric}"]=sum(d==0 for d in diffs)
                sd=statistics.stdev(diffs) if len(diffs)>1 else 0.0
                row[f"paired_effect_dz_vs_lambda0_{metric}"]=(sum(diffs)/len(diffs)/sd if sd>0 else None)
                pct=[
                    100.0*(float(arms[lam][s][metric])-float(control[s][metric]))/float(control[s][metric])
                    for s in SEEDS if float(control[s][metric]) != 0.0
                ]
                row[f"mean_paired_percent_change_vs_lambda0_{metric}"]=(sum(pct)/len(pct) if pct else None)
                row[f"median_paired_percent_change_vs_lambda0_{metric}"]=(statistics.median(pct) if pct else None)
                for seed,d in zip(SEEDS,diffs):
                    paired.append({
                        "lambda_phi":lam,"demand_seed":seed,"metric":metric,
                        "delta_vs_lambda0":d,
                    })
        table.append(row)

    OUT.mkdir(parents=True,exist_ok=True)
    fields=[]
    for r in table:
        for k in r:
            if k not in fields: fields.append(k)
    with (OUT/"lambda_sensitivity_table.csv").open("w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=fields); w.writeheader(); w.writerows(table)
    with (OUT/"lambda_paired_differences.csv").open("w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=["lambda_phi","demand_seed","metric","delta_vs_lambda0"])
        w.writeheader(); w.writerows(paired)

    md=["| Lambda phi | Credit coverage | Nonzero shaping | Pickup wait (s) | Empty km | Net energy (kWh) |",
        "|---:|---:|---:|---:|---:|---:|"]
    for r in table:
        md.append(
            f'| {r["lambda_phi"]:.2f} | {r["shapley_credit_coverage"]:.4f} | '
            f'{r["nonzero_shaping_transitions"]} | {r["mean_mean_pickup_wait_s"]:.2f} | '
            f'{r["mean_empty_km"]:.3f} | {r["mean_net_energy_kWh"]:.3f} |'
        )
    (OUT/"lambda_sensitivity_table.md").write_text("\n".join(md)+"\n",encoding="utf-8")

    x=[r["lambda_phi"] for r in table]
    for key,ylabel,name in [
        ("mean_mean_pickup_wait_s","Mean passenger pickup wait (s)","fig_lambda_pickup_wait"),
        ("mean_empty_km","Mean empty distance (km)","fig_lambda_empty_distance"),
        ("mean_net_energy_kWh","Mean net energy (kWh)","fig_lambda_energy"),
        ("shapley_credit_coverage","Shapley credit coverage","fig_lambda_credit_coverage"),
    ]:
        fig,ax=plt.subplots(figsize=(6.4,4.2))
        ax.plot(x,[r[key] for r in table],marker="o")
        ax.set_xlabel("Shapley reward coefficient, lambda phi")
        ax.set_ylabel(ylabel)
        ax.grid(True,alpha=0.25)
        fig.tight_layout()
        fig.savefig(OUT/f"{name}.png",dpi=300)
        fig.savefig(OUT/f"{name}.pdf")
        plt.close(fig)

    (OUT/"publication_summary.json").write_text(json.dumps({
        "table":table,
        "paired_against_lambda_zero":True,
        "heldout_seeds":list(SEEDS),
        "statistical_claim":"descriptive paired multi-seed ablation; no inferential significance test is claimed",
        "direction":"lower is better for all operational outcome metrics",
        "interpretation":"Paired CRN ablation; lambda=0 is the DCF control without Shapley reward shaping. Paired dz is descriptive. No grid value is declared globally optimal."
    },indent=2))
    print(json.dumps(table,indent=2))

if __name__=="__main__":
    main()
