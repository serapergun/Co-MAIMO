"""TraCI runner for externally dispatched SUMO taxi experiments."""
from pathlib import Path
import csv
import json
import shutil

def run(sumocfg, out_dir, dispatch_fn, max_time_s=None):
    import traci

    binary = shutil.which("sumo")
    if not binary:
        raise RuntimeError("SUMO binary not found on PATH.")
    cfg = Path(sumocfg)
    if not cfg.exists():
        raise FileNotFoundError(cfg)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tripinfo = out / "tripinfo.xml"
    summary = out / "summary.xml"
    emissions = out / "emissions.xml"

    cmd = [
        binary, "-c", str(cfg),
        "--device.taxi.dispatch-algorithm", "traci",
        "--tripinfo-output", str(tripinfo),
        "--summary-output", str(summary),
        "--emission-output", str(emissions),
        "--emission-output.step-scaled", "true",
        "--tripinfo-output.write-unfinished", "true",
        "--no-step-log", "true",
    ]

    rows = []
    assigned_reservations = set()
    duplicate_assignments = set()
    teleports = set()
    traci.start(cmd)
    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            now = float(traci.simulation.getTime())
            if max_time_s is not None and now > max_time_s:
                raise RuntimeError("Simulation exceeded horizon.")

            teleports.update(traci.simulation.getStartingTeleportIDList())
            new_reservations = traci.person.getTaxiReservations(1)
            active_reservations = traci.person.getTaxiReservations(0)
            fleet = traci.vehicle.getTaxiFleet(-1)
            assignments = dispatch_fn(traci, new_reservations, fleet, now) or []
            for vehicle_id, reservation_ids in assignments:
                for reservation_id in reservation_ids:
                    if reservation_id in assigned_reservations:
                        duplicate_assignments.add(reservation_id)
                    assigned_reservations.add(reservation_id)
                traci.vehicle.dispatchTaxi(vehicle_id, reservation_ids)

            rows.append({
                "time": now,
                "active_reservations": len(active_reservations),
                "idle_taxis": len(traci.vehicle.getTaxiFleet(0)),
                "pickup_taxis": len(traci.vehicle.getTaxiFleet(1)),
                "occupied_taxis": len(traci.vehicle.getTaxiFleet(2)),
                "fleet_visible": len(fleet),
                "min_expected": traci.simulation.getMinExpectedNumber(),
            })
    finally:
        final_reservations = list(traci.person.getTaxiReservations(0))
        remaining_persons = list(traci.person.getIDList())
        traci.close()

    with (out / "system_log.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else ["time"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    qc = {
        "assigned_reservations": sorted(assigned_reservations),
        "duplicate_assignments": sorted(duplicate_assignments),
        "final_active_reservations": [r.id for r in final_reservations],
        "remaining_persons": remaining_persons,
        "teleported_vehicle_ids": sorted(teleports),
    }
    (out / "run_command.json").write_text(json.dumps(cmd, indent=2), encoding="utf-8")
    (out / "request_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    return {
        "tripinfo": str(tripinfo),
        "summary": str(summary),
        "emissions": str(emissions),
        "request_qc": str(out / "request_qc.json"),
    }
