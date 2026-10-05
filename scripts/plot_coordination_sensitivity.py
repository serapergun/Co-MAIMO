"""Publication artifacts for DCF coordination-overhead sensitivity."""
from pathlib import Path
import csv, json
import matplotlib.pyplot as plt

ROOT=Path("artifacts/barcelona_coordination_sensitivity")
ROWS=ROOT/"sensitivity_rows.json"
OUT=ROOT/"publication"

def mean(xs):
    return sum(xs)/len(xs) if xs else None

def main():
    rows=json.loads(ROWS.read_text())
    if not rows:
        raise SystemExit("no coordination sensitivity rows")
    overheads=sorted({float(r["coordination_overhead_s"]) for r in rows})
    expected={0.0,5.0,15.0,30.0}
    if set(overheads)!=expected:
        raise SystemExit(f"unexpected overhead grid: {overheads}")
    table=[]
    all_thresholds=[]
    for overhead in overheads:
        subset=[r for r in rows if float(r["coordination_overhead_s"])==overhead]
        seeds={int(r["demand_seed"]) for r in subset}
        if seeds!={9101,9102,9103,9104,9105}:
            raise SystemExit(f"incomplete paired CRN rows for overhead {overhead}: {seeds}")
        events=[]
        for seed in sorted(seeds):
            path=ROOT/f"overhead_{int(overhead)}"/f"seed_{seed}"/"coalition_events.json"
            events.extend(json.loads(path.read_text()))
        thresholds=[float(e["effective_break_even_overhead_s"]) for e in events]
        all_thresholds.extend(thresholds)
        table.append({
            "coordination_overhead_s":overhead,
            "mean_acceptance_rate":mean([float(r["acceptance_rate"]) for r in subset]),
            "mean_accepted_coalitions":mean([float(r["accepted_coalitions"]) for r in subset]),
            "mean_pickup_wait_s":mean([float(r["mean_pickup_wait_s"]) for r in subset]),
            "mean_empty_km":mean([float(r["empty_km"]) for r in subset]),
            "mean_net_energy_kWh":mean([float(r["net_energy_kWh"]) for r in subset]),
            "mean_energy_intensity_kWh_per_vehicle_km":mean([float(r["kWh_per_vehicle_km"]) for r in subset]),
            "proposal_count":len(events),
            "mean_effective_break_even_overhead_s":mean(thresholds),
        })
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"coordination_sensitivity_table.csv").open("w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    md=["| Overhead (s) | Acceptance rate | Accepted coalitions | Pickup wait (s) | Empty km | Net energy (kWh) |",
        "|---:|---:|---:|---:|---:|---:|"]
    for r in table:
        md.append(f'| {r["coordination_overhead_s"]:.0f} | {r["mean_acceptance_rate"]:.4f} | {r["mean_accepted_coalitions"]:.2f} | {r["mean_pickup_wait_s"]:.2f} | {r["mean_empty_km"]:.3f} | {r["mean_net_energy_kWh"]:.3f} |')
    (OUT/"coordination_sensitivity_table.md").write_text("\n".join(md)+"\n",encoding="utf-8")

    x=[r["coordination_overhead_s"] for r in table]
    for key,ylabel,name in [
        ("mean_acceptance_rate","Mean coalition acceptance rate","fig_coordination_acceptance"),
        ("mean_pickup_wait_s","Mean passenger pickup wait (s)","fig_coordination_pickup_wait"),
        ("mean_net_energy_kWh","Mean net energy (kWh)","fig_coordination_energy"),
    ]:
        fig,ax=plt.subplots(figsize=(6.4,4.2))
        ax.plot(x,[r[key] for r in table],marker="o")
        ax.set_xlabel("Coordination overhead (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True,alpha=0.25)
        fig.tight_layout()
        fig.savefig(OUT/f"{name}.png",dpi=300)
        fig.savefig(OUT/f"{name}.pdf")
        plt.close(fig)

    fig,ax=plt.subplots(figsize=(6.4,4.2))
    ax.hist(all_thresholds,bins=min(20,max(5,len(all_thresholds)//5)))
    ax.set_xlabel("Effective break-even coordination overhead (s)")
    ax.set_ylabel("Coalition proposals")
    ax.grid(True,alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT/"fig_coordination_break_even_distribution.png",dpi=300)
    fig.savefig(OUT/"fig_coordination_break_even_distribution.pdf")
    plt.close(fig)
    (OUT/"publication_summary.json").write_text(json.dumps({
        "table":table,
        "break_even_proposal_count":len(all_thresholds),
        "interpretation":"Controlled overhead sensitivity; overhead values are scenarios, while break-even thresholds are derived from SUMO route-time counterfactuals."
    },indent=2))
    print(json.dumps(table,indent=2))

if __name__=="__main__":
    main()
