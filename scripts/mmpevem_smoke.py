from pathlib import Path
import os, subprocess
import sumo

root=Path("artifacts/mmpevem_smoke")
root.mkdir(parents=True,exist_ok=True)
net=root/"network.net.xml"
rou=root/"routes.rou.xml"
cfg=root/"run.sumocfg"
trip=root/"tripinfo.xml"

nodes=root/"nodes.nod.xml"
edges=root/"edges.edg.xml"
nodes.write_text("""<nodes>
<n id="n0" x="0" y="0"/><n id="n1" x="1000" y="0"/><n id="n2" x="2000" y="0"/>
</nodes>""")
edges.write_text("""<edges>
<edge id="e0" from="n0" to="n1" numLanes="1" speed="13.89"/>
<edge id="e1" from="n1" to="n2" numLanes="1" speed="13.89"/>
</edges>""")
rou.write_text("""<routes>
<vType id="ev" vClass="passenger" emissionClass="MMPEVEM" accel="2.0" decel="4.5" maxSpeed="13.89">
 <param key="has.battery.device" value="true"/>
</vType>
<route id="r0" edges="e0 e1"/>
<vehicle id="ev0" type="ev" route="r0" depart="0"/>
</routes>""")
cfg.write_text(f"""<configuration><input>
<net-file value="{net.name}"/><route-files value="{rou.name}"/>
</input><time><begin value="0"/><end value="600"/></time></configuration>""")

bin_dir=Path(sumo.SUMO_HOME)/"bin"
netconvert=bin_dir/"netconvert"
sumo_bin=bin_dir/"sumo"
subprocess.run([str(netconvert),"--node-files",nodes.name,"--edge-files",edges.name,"--output-file",net.name],check=True,cwd=root)
subprocess.run([str(sumo_bin),"-c",cfg.name,"--tripinfo-output",trip.name,"--tripinfo-output.write-unfinished","true","--no-step-log","true"],check=True,cwd=root)
if not trip.exists():
    raise SystemExit("tripinfo was not produced")
print(trip)
