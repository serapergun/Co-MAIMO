"""True B1 TraCI taxi lifecycle fixture on the Barcelona preflight network.

Integration validation only: one deterministic synthetic request, not EMEF demand.
"""
from pathlib import Path
import json
import sumolib

from src.baselines.b1_dispatch import dispatch
from src.simulation.traci_runner import run

ROOT = Path("artifacts/barcelona_b1_traci")
ROOT.mkdir(parents=True, exist_ok=True)
NET = Path("artifacts/barcelona_network/barcelona.net.xml")
if not NET.exists():
    raise SystemExit("Barcelona preflight network missing")

net = sumolib.net.readNet(str(NET))
edges = [e for e in net.getEdges()
         if not e.getID().startswith(":")
         and any(l.allows("passenger") for l in e.getLanes())]

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

origin, destination = chosen
rou = ROOT / "b1.rou.xml"
cfg = ROOT / "b1.sumocfg"

rou.write_text(f"""<routes>
<vType id="evTaxi" vClass="passenger" emissionClass="MMPEVEM"
       personCapacity="4" accel="2.0" decel="4.5" maxSpeed="13.89">
  <param key="has.battery.device" value="true"/>
  <param key="has.taxi.device" value="true"/>
</vType>
<vehicle id="taxi0" type="evTaxi" depart="0" line="taxi">
  <route edges="{origin}"/>
</vehicle>
<person id="request0" depart="10">
  <ride from="{origin}" to="{destination}" lines="taxi"/>
</person>
</routes>""", encoding="utf-8")

cfg.write_text(f"""<configuration>
<input>
  <net-file value="{NET.resolve()}"/>
  <route-files value="{rou.resolve()}"/>
</input>
<time><begin value="0"/><end value="3600"/><step-length value="1"/></time>
</configuration>""", encoding="utf-8")

outputs = run(cfg, ROOT, dispatch, max_time_s=3600)
(ROOT / "fixture_manifest.json").write_text(json.dumps({
    "role": "B1 TraCI lifecycle integration fixture; deterministic synthetic request; not EMEF demand",
    "origin_edge": origin,
    "destination_edge": destination,
    "request_id": "request0",
    "taxi_id": "taxi0",
    "outputs": outputs,
}, indent=2), encoding="utf-8")
print(json.dumps(outputs, indent=2))
