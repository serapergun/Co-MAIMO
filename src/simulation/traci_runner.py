from pathlib import Path
import csv,json,shutil
def run(sumocfg,out_dir,dispatch_fn,max_time_s=None):
 import traci
 binary=shutil.which("sumo")
 if not binary: raise RuntimeError("SUMO binary not found on PATH.")
 cfg=Path(sumocfg)
 if not cfg.exists(): raise FileNotFoundError(cfg)
 out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
 tripinfo=out/"tripinfo.xml"; summary=out/"summary.xml"
 cmd=[binary,"-c",str(cfg),"--device.taxi.dispatch-algorithm","traci","--tripinfo-output",str(tripinfo),"--summary-output",str(summary),"--tripinfo-output.write-unfinished","true"]
 rows=[]; traci.start(cmd)
 try:
  while traci.simulation.getMinExpectedNumber()>0:
   traci.simulationStep(); now=float(traci.simulation.getTime())
   if max_time_s is not None and now>max_time_s: raise RuntimeError("Simulation exceeded horizon.")
   reservations=traci.person.getTaxiReservations(0); fleet=traci.vehicle.getTaxiFleet(-1)
   for veh,res_ids in (dispatch_fn(traci,reservations,fleet,now) or []): traci.vehicle.dispatchTaxi(veh,res_ids)
   rows.append({"time":now,"pending_reservations":len(reservations),"fleet_visible":len(fleet),"min_expected":traci.simulation.getMinExpectedNumber()})
 finally: traci.close()
 with (out/"system_log.csv").open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys() if rows else ["time"]); w.writeheader(); w.writerows(rows)
 (out/"run_command.json").write_text(json.dumps(cmd,indent=2),encoding="utf-8")
 return {"tripinfo":str(tripinfo),"summary":str(summary)}
