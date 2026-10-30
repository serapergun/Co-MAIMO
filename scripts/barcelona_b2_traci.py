"""B2 TraCI fixture: two taxis and two deterministic simultaneous requests.

Integration validation only; not calibrated EMEF demand.
"""
from pathlib import Path
import json
import sumolib

from src.baselines.b1_dispatch import dispatch
from src.simulation.traci_runner import run

ROOT=Path("artifacts/barcelona_b2_traci"); ROOT.mkdir(parents=True,exist_ok=True)
NET=Path("artifacts/barcelona_network/barcelona.net.xml")
if not NET.exists(): raise SystemExit("Barcelona preflight network missing")
net=sumolib.net.readNet(str(NET))
edges=[e for e in net.getEdges() if not e.getID().startswith(":") and any(l.allows("passenger") for l in e.getLanes())]

pairs=[]
for o in edges[:1500]:
    for d in edges[-1:max(-1501,-len(edges)):-25]:
        if o.getID()==d.getID(): continue
        route,_=net.getOptimalPath(o,d,vClass="passenger")
        if route and sum(e.getLength() for e in route)>=3000:
            p=(o.getID(),d.getID())
            if not pairs or p[0]!=pairs[0][0]:
                pairs.append(p)
                if len(pairs)==2: break
    if len(pairs)==2: break
if len(pairs)!=2: raise SystemExit("Could not find two deterministic >=3 km routes")

(o0,d0),(o1,d1)=pairs
rou=ROOT/"b2.rou.xml"; cfg=ROOT/"b2.sumocfg"
rou.write_text(f"""<routes>
<vType id="evTaxi" vClass="passenger" emissionClass="MMPEVEM" personCapacity="4" accel="2.0" decel="4.5" maxSpeed="13.89">
  <param key="has.battery.device" value="true"/><param key="has.taxi.device" value="true"/>
</vType>
<vehicle id="taxi0" type="evTaxi" depart="0" line="taxi"><route edges="{o0}"/></vehicle>
<vehicle id="taxi1" type="evTaxi" depart="0" line="taxi"><route edges="{o1}"/></vehicle>
<person id="request0" depart="10"><ride from="{o0}" to="{d0}" lines="taxi"/></person>
<person id="request1" depart="10"><ride from="{o1}" to="{d1}" lines="taxi"/></person>
</routes>""",encoding="utf-8")
cfg.write_text(f"""<configuration><input><net-file value="{NET.resolve()}"/><route-files value="{rou.resolve()}"/></input><time><begin value="0"/><end value="3600"/><step-length value="1"/></time></configuration>""",encoding="utf-8")
outputs=run(cfg,ROOT,dispatch,max_time_s=3600)
(ROOT/"fixture_manifest.json").write_text(json.dumps({"role":"B2 integration fixture; two taxis/two deterministic requests; not EMEF demand","pairs":pairs,"outputs":outputs},indent=2),encoding="utf-8")
print(json.dumps(outputs,indent=2))
