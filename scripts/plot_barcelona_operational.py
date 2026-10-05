"""Create publication-ready operational sensitivity figures from aggregate CSV."""
from pathlib import Path
import csv
import matplotlib.pyplot as plt

BASE=Path("artifacts/barcelona_operational/aggregate")
rows=[]
with (BASE/"operational_matrix.csv").open(encoding="utf-8") as f:
    for r in csv.DictReader(f):
        for k in ("taxis","requests"): r[k]=int(r[k])
        for k in ("mean_assignment_wait_s","mean_pickup_wait_s","empty_km_share","net_energy_kWh","fleet_vehicle_km","kWh_per_vehicle_km"):
            r[k]=float(r[k])
        rows.append(r)

def plot_metric(metric, ylabel, filename, scale=1.0):
    fig,ax=plt.subplots(figsize=(6.4,4.2))
    for req in (12,24,36):
        pts=sorted((r for r in rows if r["requests"]==req), key=lambda r:r["taxis"])
        ax.plot([r["taxis"] for r in pts],[scale*r[metric] for r in pts],marker="o",label=f"{req} requests")
    ax.set_xlabel("Fleet size (taxis)")
    ax.set_ylabel(ylabel)
    ax.set_xticks([4,8,12])
    ax.grid(True,alpha=.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(BASE/filename,dpi=300,bbox_inches="tight")
    fig.savefig(BASE/filename.replace(".png",".pdf"),bbox_inches="tight")
    plt.close(fig)

plot_metric("mean_assignment_wait_s","Mean assignment delay (s)","fig_assignment_delay.png")
plot_metric("mean_pickup_wait_s","Mean passenger pickup wait (s)","fig_pickup_wait.png")
plot_metric("empty_km_share","Empty-distance share (%)","fig_empty_distance_share.png",100)
plot_metric("net_energy_kWh","Net fleet energy (kWh)","fig_net_energy.png")
plot_metric("kWh_per_vehicle_km","Energy intensity (kWh/vehicle-km)","fig_energy_intensity.png")
print("Wrote 5 PNG and 5 PDF operational sensitivity figures")
