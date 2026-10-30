"""Aggregate Barcelona operational matrix outputs into reproducible tables."""
from pathlib import Path
import csv, json

BASE = Path("artifacts/barcelona_operational")
OUT = BASE / "aggregate"
OUT.mkdir(parents=True, exist_ok=True)

rows = []
for taxis in (4, 8, 12):
    for requests in (12, 24, 36):
        p = BASE / f"t{taxis}_r{requests}" / "operational_metrics.json"
        if not p.exists():
            raise SystemExit(f"missing matrix result: {p}")
        r = json.loads(p.read_text(encoding="utf-8"))
        if r["completed"] != requests or r["unresolved"] != 0:
            raise RuntimeError(f"incomplete cell t{taxis}_r{requests}")
        if r["teleports"] != 0 or r["horizon_exceeded"]:
            raise RuntimeError(f"invalid QC cell t{taxis}_r{requests}")
        if abs(r["net_energy_kWh"] - (r["gross_energy_kWh"] - r["regenerated_energy_kWh"])) > 1e-9:
            raise RuntimeError(f"energy identity failed t{taxis}_r{requests}")
        rows.append(r)

fields = [
    "taxis","requests","assigned","completed","unresolved","teleports",
    "horizon_exceeded","final_simulation_time_s","mean_assignment_wait_s",
    "max_assignment_wait_s","mean_pickup_wait_s","max_pickup_wait_s",
    "mean_post_assignment_pickup_s","occupied_km","empty_km","empty_km_share",
    "fleet_vehicle_km","gross_energy_kWh","regenerated_energy_kWh","net_energy_kWh",
    "kWh_per_vehicle_km",
]
with (OUT / "operational_matrix.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k) for k in fields})

(OUT / "operational_matrix.json").write_text(
    json.dumps(rows, indent=2), encoding="utf-8"
)

md = [
    "| Taxis | Requests | Completed | Mean assignment delay (s) | Mean pickup wait (s) | Max pickup wait (s) | Empty share (%) | Fleet km | Net energy (kWh) | kWh/veh-km |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    md.append(
        f'| {r["taxis"]} | {r["requests"]} | {r["completed"]}/{r["requests"]} | '
        f'{r["mean_assignment_wait_s"]:.1f} | {r["mean_pickup_wait_s"]:.1f} | '
        f'{r["max_pickup_wait_s"]:.1f} | {100*r["empty_km_share"]:.2f} | '
        f'{r["fleet_vehicle_km"]:.3f} | {r["net_energy_kWh"]:.3f} | '
        f'{r["kWh_per_vehicle_km"]:.5f} |'
    )
(OUT / "operational_matrix_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
