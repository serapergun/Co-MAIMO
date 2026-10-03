"""B4 TraCI load fixture: eight taxis and twenty-four deterministic staggered requests.

Integration/stress validation only; not calibrated EMEF demand.
"""
from pathlib import Path
import json
import sumolib
from src.baselines.b1_dispatch import dispatch
from src.simulation.traci_runner import run

N_TAXIS=8
N_REQUESTS=24
ROOT=Path("artifacts/barcelona_b4_traci"); ROOT.mkdir(parents=True,exist_ok=True)
NET=Path("artifacts/barcelona_network/barcelona.net.xml")
if not NET.exists(): raise SystemExit("Barcelona preflight network missing")
net=sumolib.net.readNet(str(NET))
edges=[e for e in net.getEdges() if not e.getID().startswith(":") and any(l.allows("passenger") for l in e.getLanes())]

pairs=[]
used_origins=set()
for o in edges[:2000]:
    if o.getID() in used_origins: continue
    for d in edges[-1:max(-2001,-len(edges)):-20]:
        if o.getID()==d.getID(): continue
        route,_=net.getOptimalPath(o,d,vClass="passenger")
        if not route:
            continue
        route_len=sum(e.getLength() for e in route)
        # Avoid fragile seeds: require a moderate connected route and enough
        # usable lane length at both endpoints for stable taxi insertion/pickup.
        o_lanes=[l for l in o.getLanes() if l.allows("passenger")]
        d_lanes=[l for l in d.getLanes() if l.allows("passenger")]
        if (3000 <= route_len <= 6000
                and max((l.getLength() for l in o_lanes), default=0) >= 80
                and max((l.getLength() for l in d_lanes), default=0) >= 80):
            pairs.append((o.getID(),d.getID()))
            used_origins.add(o.getID())
            break
    if len(pairs)==N_TAXIS: break
if len(pairs)!=N_TAXIS: raise SystemExit(f"Could not find {N_TAXIS} deterministic >=3 km routes")

routes=['<routes>','<vType id="evTaxi" vClass="passenger" emissionClass="MMPEVEM" personCapacity="4" accel="2.0" decel="4.5" maxSpeed="13.89"><param key="has.battery.device" value="true"/><param key="has.taxi.device" value="true"/></vType>']
for i,(o,_) in enumerate(pairs):
    routes.append(f'<vehicle id="taxi{i}" type="evTaxi" depart="0" line="taxi"><route edges="{o}"/></vehicle>')
# Two waves. Each wave has one request originating at each taxi seed edge.
for wave,depart in enumerate((10,1800,3600)):
    for i,(o,d) in enumerate(pairs):
        rid=wave*N_TAXIS+i
        routes.append(f'<person id="request{rid}" depart="{depart}"><ride from="{o}" to="{d}" lines="taxi"/></person>')
routes.append('</routes>')
rou=ROOT/"b4.rou.xml"; cfg=ROOT/"b4.sumocfg"
rou.write_text("\n".join(routes),encoding="utf-8")
cfg.write_text(f'<configuration><input><net-file value="{NET.resolve()}"/><route-files value="{rou.resolve()}"/></input><time><begin value="0"/><end value="18000"/><step-length value="1"/></time></configuration>',encoding="utf-8")
outputs=run(cfg,ROOT,dispatch,max_time_s=18000)
(ROOT/"fixture_manifest.json").write_text(json.dumps({"role":"B4 load integration fixture; eight taxis/twenty-four staggered deterministic requests; not EMEF demand","taxi_count":N_TAXIS,"request_count":N_REQUESTS,"pairs":pairs,"departures":[10,1800,3600],"outputs":outputs},indent=2),encoding="utf-8")
print(json.dumps(outputs,indent=2))
