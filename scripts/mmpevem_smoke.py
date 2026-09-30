from pathlib import Path
import subprocess
import sumo

root = Path("artifacts/mmpevem_smoke")
root.mkdir(parents=True, exist_ok=True)
net = root / "network.net.xml"
rou = root / "routes.rou.xml"
cfg = root / "run.sumocfg"
trip = root / "tripinfo.xml"
emissions = root / "emissions.xml"
nodes = root / "nodes.nod.xml"
edges = root / "edges.edg.xml"

nodes.write_text("""<nodes>
<node id="n0" x="0" y="0"/>
<node id="n1" x="1000" y="0"/>
<node id="n2" x="2000" y="0"/>
</nodes>""", encoding="utf-8")
edges.write_text("""<edges>
<edge id="e0" from="n0" to="n1" numLanes="1" speed="13.89"/>
<edge id="e1" from="n1" to="n2" numLanes="1" speed="13.89"/>
</edges>""", encoding="utf-8")
rou.write_text("""<routes>
<vType id="ev" vClass="passenger" emissionClass="MMPEVEM" accel="2.0" decel="4.5" maxSpeed="13.89">
  <param key="has.battery.device" value="true"/>
</vType>
<route id="r0" edges="e0 e1"/>
<vehicle id="ev0" type="ev" route="r0" depart="0"/>
</routes>""", encoding="utf-8")
cfg.write_text("""<configuration>
<input>
  <net-file value="network.net.xml"/>
  <route-files value="routes.rou.xml"/>
</input>
<time><begin value="0"/><end value="600"/><step-length value="1"/></time>
</configuration>""", encoding="utf-8")

bin_dir = Path(sumo.SUMO_HOME) / "bin"
subprocess.run([str(bin_dir / "netconvert"), "--node-files", nodes.name,
                "--edge-files", edges.name, "--output-file", net.name],
               check=True, cwd=root)
subprocess.run([str(bin_dir / "sumo"), "-c", cfg.name,
                "--tripinfo-output", trip.name,
                "--tripinfo-output.write-unfinished", "true",
                "--emission-output", emissions.name,
                "--emission-output.step-scaled", "true",
                "--no-step-log", "true"],
               check=True, cwd=root)

for output in (trip, emissions):
    if not output.exists() or output.stat().st_size == 0:
        raise SystemExit(f"required SUMO output missing or empty: {output}")
print(trip)
print(emissions)
