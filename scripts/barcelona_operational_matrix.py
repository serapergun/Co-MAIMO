"""Deterministic Barcelona operational fleet-size x demand-load experiment."""
from pathlib import Path
import argparse, json, xml.etree.ElementTree as ET
import sumolib
from src.baselines.b1_dispatch import dispatch
from src.simulation.traci_runner import run
from src.evaluation.parse_mmpevem_emissions import mmpevem_emission_energy

p=argparse.ArgumentParser(); p.add_argument("--taxis",type=int,required=True); p.add_argument("--requests",type=int,required=True); a=p.parse_args()
if a.taxis not in (4,8,12) or a.requests not in (12,24,36): raise SystemExit("unsupported matrix cell")
root=Path("artifacts/barcelona_operational")/f"t{a.taxis}_r{a.requests}"; root.mkdir(parents=True,exist_ok=True)
net_path=Path("artifacts/barcelona_network/barcelona.net.xml")
if not net_path.exists(): raise SystemExit("Barcelona preflight network missing")
net=sumolib.net.readNet(str(net_path))
edges=[e for e in net.getEdges() if not e.getID().startswith(":") and any(l.allows("passenger") for l in e.getLanes())]
pairs=[]; used=set()
# Corridors reproducibly observed to trigger SUMO wrong-lane teleport recovery
# in this fixed Barcelona snapshot. Exclude any OD whose optimal passenger path
# traverses them; the same cleaned OD pool is used for every matrix cell.
FRAGILE_EDGE_PREFIXES=(
    "87504601#3", "-306281847#2", "-545747621", "-545021320#3",
    "-545747622#3", "-753154730#3", "-285182795", "533269953",
    "300828583#0",
)
for o in edges[:3000]:
    if o.getID() in used: continue
    for d in edges[-1:max(-3001,-len(edges)):-20]:
        if o.getID()==d.getID(): continue
        route,_=net.getOptimalPath(o,d,vClass="passenger")
        if not route: continue
        path_edges=list(route)
        L=sum(e.getLength() for e in path_edges)
        ol=[l for l in o.getLanes() if l.allows("passenger")]
        dl=[l for l in d.getLanes() if l.allows("passenger")]
        min_edge_len=min((e.getLength() for e in path_edges), default=0)
        all_have_passenger_lane=all(
            any(l.allows("passenger") for l in e.getLanes()) for e in path_edges
        )
        avoids_fragile_corridors=all(
            not any(e.getID().startswith(prefix) for prefix in FRAGILE_EDGE_PREFIXES)
            for e in path_edges
        )
        if (3000<=L<=6000
                and max((l.getLength() for l in ol),default=0)>=80
                and max((l.getLength() for l in dl),default=0)>=80
                and min_edge_len>=25
                and all_have_passenger_lane
                and avoids_fragile_corridors):
            pairs.append((o.getID(),d.getID())); used.add(o.getID()); break
    if len(pairs)==12: break
if len(pairs)<12: raise SystemExit("Could not construct 12 safe deterministic OD seeds")

routes=['<routes>','<vType id="evTaxi" vClass="passenger" emissionClass="MMPEVEM" personCapacity="4" accel="2.0" decel="4.5" maxSpeed="13.89"><param key="has.battery.device" value="true"/><param key="has.taxi.device" value="true"/></vType>']
for i in range(a.taxis):
    routes.append(f'<vehicle id="taxi{i}" type="evTaxi" depart="0" line="taxi"><route edges="{pairs[i][0]}"/></vehicle>')
# 12 requests per wave, preserving identical OD/departure demand across fleet-size comparisons.
for rid in range(a.requests):
    wave=rid//12; i=rid%12; depart=10+1800*wave
    o,d=pairs[i]
    routes.append(f'<person id="request{rid}" depart="{depart}"><ride from="{o}" to="{d}" lines="taxi"/></person>')
routes.append('</routes>')
rou=root/"scenario.rou.xml"; cfg=root/"scenario.sumocfg"
rou.write_text("\n".join(routes),encoding="utf-8")
cfg.write_text(f'<configuration><input><net-file value="{net_path.resolve()}"/><route-files value="{rou.resolve()}"/></input><time><begin value="0"/><end value="18000"/><step-length value="1"/></time></configuration>',encoding="utf-8")
outputs=run(cfg,root,dispatch,max_time_s=18000)
q=json.loads((root/"request_qc.json").read_text())
trips=ET.parse(root/"tripinfo.xml").getroot().findall("tripinfo")
taxi_ids={f"taxi{i}" for i in range(a.taxis)}
fleet_km=sum(float(t.get("routeLength","0")) for t in trips if t.get("id") in taxi_ids)/1000
m=mmpevem_emission_energy(root/"emissions.xml")
occ=sum(q["occupied_distance_m_by_taxi"].values())/1000
empty=sum(q["empty_distance_m_by_taxi"].values())/1000
wait=list(q["assignment_wait_s"].values())
summary={
 "taxis":a.taxis,"requests":a.requests,"assigned":len(q["assigned_reservations"]),
 "completed":a.requests-len(q["final_active_reservations"]),"unresolved":len(q["final_active_reservations"]),
 "teleports":len(q["teleported_vehicle_ids"]),"horizon_exceeded":q["horizon_exceeded"],
 "final_simulation_time_s":q["final_simulation_time_s"],
 "mean_assignment_wait_s":sum(wait)/len(wait) if wait else None,"max_assignment_wait_s":max(wait) if wait else None,
 "occupied_km":occ,"empty_km":empty,"empty_km_share":empty/(empty+occ) if empty+occ else None,
 "fleet_vehicle_km":fleet_km,**m}
summary["kWh_per_vehicle_km"]=m["net_energy_kWh"]/fleet_km if fleet_km else None
(root/"operational_metrics.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(json.dumps(summary,indent=2))
# Teleport recovery invalidates operational distance/time/energy comparison.
if summary["teleports"] != 0:
    raise RuntimeError(f'Operational QC failed: {summary["teleports"]} teleported taxi(s)')
