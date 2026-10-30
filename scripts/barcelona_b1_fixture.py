"""Deterministic Barcelona-network MMPEVEM movement fixture.

This validates routability and physical-energy output on the Barcelona preflight
network. It is deliberately NOT a taxi-dispatch/B1 or EMEF-calibrated experiment.
"""
from pathlib import Path
import json
import shutil
import subprocess
import sumolib

ROOT = Path("artifacts/barcelona_b1_fixture")
ROOT.mkdir(parents=True, exist_ok=True)
NET = Path("artifacts/barcelona_network/barcelona.net.xml")
if not NET.exists():
    raise SystemExit("Barcelona preflight network missing")

net = sumolib.net.readNet(str(NET))
edges = [
    e for e in net.getEdges()
    if not e.getID().startswith(":")
    and any(l.allows("passenger") for l in e.getLanes())
]

sumo_bin = shutil.which("sumo")
if not sumo_bin:
    raise SystemExit("sumo not on PATH")

# Deterministically select a feasible route of at least 3 km.
chosen = None
for o in edges[:1500]:
    for d in edges[-1:max(-1501, -len(edges)):-25]:
        if o.getID() == d.getID():
            continue
        route, _ = net.getOptimalPath(o, d, vClass="passenger")
        if route and sum(e.getLength() for e in route) >= 3000:
            chosen = (o.getID(), d.getID())
            break
    if chosen:
        break
if not chosen:
    raise SystemExit("No deterministic >=3 km passenger route found")

o, d = chosen
rou = ROOT / "fixture.rou.xml"
cfg = ROOT / "fixture.sumocfg"
trip = ROOT / "tripinfo.xml"
em = ROOT / "emissions.xml"
summary = ROOT / "summary.xml"

# Important: this movement fixture must not enable the taxi device. An idle taxi
# without a reservation intentionally waits for dispatch and therefore does not
# validate ordinary network traversal.
rou.write_text(f"""<routes>
<vType id="ev" vClass="passenger" emissionClass="MMPEVEM" accel="2.0" decel="4.5" maxSpeed="13.89">
 <param key="has.battery.device" value="true"/>
</vType>
<trip id="ev0" type="ev" from="{o}" to="{d}" depart="0"/>
</routes>""", encoding="utf-8")

cfg.write_text(f"""<configuration>
<input>
 <net-file value="{NET.resolve()}"/>
 <route-files value="{rou.resolve()}"/>
</input>
<time><begin value="0"/><end value="3600"/><step-length value="1"/></time>
</configuration>""", encoding="utf-8")

cmd = [
    sumo_bin, "-c", str(cfg),
    "--tripinfo-output", str(trip),
    "--emission-output", str(em),
    "--emission-output.step-scaled", "true",
    "--summary-output", str(summary),
    "--tripinfo-output.write-unfinished", "true",
    "--no-step-log", "true",
    "--no-warnings", "true",
]
subprocess.run(cmd, check=True)

for p in (trip, em, summary):
    if not p.exists() or p.stat().st_size == 0:
        raise SystemExit(f"missing/empty output: {p}")

(ROOT / "fixture_manifest.json").write_text(json.dumps({
    "role": "Barcelona network + MMPEVEM movement integration fixture; not B1; not EMEF demand",
    "origin_edge": o,
    "destination_edge": d,
    "sumo_command": cmd,
}, indent=2), encoding="utf-8")
print(o, d)
