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
    pickup_taxi_ids = set()
    occupied_taxi_ids = set()
    occupied_distance_m = {}
    empty_distance_m = {}
    previous_distance_m = {}
    request_first_seen_s = {}
    request_assigned_s = {}
    horizon_exceeded = False
    traci.start(cmd)
    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            now = float(traci.simulation.getTime())
            if max_time_s is not None and now > max_time_s:
                horizon_exceeded = True
                break

            teleports.update(traci.simulation.getStartingTeleportIDList())
            new_reservations = traci.person.getTaxiReservations(1)
            active_reservations = traci.person.getTaxiReservations(0)
            # Retry active reservations that have never been assigned. A request
            # may surface when no feasible idle taxi route exists, so onlyNew=1
            # alone can strand it forever.
            for reservation in active_reservations:
                request_first_seen_s.setdefault(reservation.id, now)
            dispatchable_reservations = [
                r for r in active_reservations
                if r.id not in assigned_reservations
            ]
            fleet = traci.vehicle.getTaxiFleet(-1)
            pickup_now = set(traci.vehicle.getTaxiFleet(1))
            occupied_now = set(traci.vehicle.getTaxiFleet(2))
            pickup_taxi_ids.update(pickup_now)
            occupied_taxi_ids.update(occupied_now)
            for vehicle_id in fleet:
                try:
                    distance_m = float(traci.vehicle.getDistance(vehicle_id))
                except Exception:
                    continue
                previous = previous_distance_m.get(vehicle_id, distance_m)
                delta = max(0.0, distance_m - previous)
                if vehicle_id in occupied_now:
                    occupied_distance_m[vehicle_id] = occupied_distance_m.get(vehicle_id, 0.0) + delta
                else:
                    empty_distance_m[vehicle_id] = empty_distance_m.get(vehicle_id, 0.0) + delta
                previous_distance_m[vehicle_id] = distance_m
            assignments = dispatch_fn(traci, dispatchable_reservations, fleet, now) or []
            for vehicle_id, reservation_ids in assignments:
                for reservation_id in reservation_ids:
                    if reservation_id in assigned_reservations:
                        duplicate_assignments.add(reservation_id)
                    assigned_reservations.add(reservation_id)
                    request_assigned_s.setdefault(reservation_id, now)
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
        "pickup_taxi_ids": sorted(pickup_taxi_ids),
        "occupied_taxi_ids": sorted(occupied_taxi_ids),
        "horizon_exceeded": horizon_exceeded,
        "final_simulation_time_s": rows[-1]["time"] if rows else 0.0,
        "occupied_distance_m_by_taxi": {
            vehicle_id: round(distance_m, 6)
            for vehicle_id, distance_m in sorted(occupied_distance_m.items())
        },
        "empty_distance_m_by_taxi": {
            vehicle_id: round(distance_m, 6)
            for vehicle_id, distance_m in sorted(empty_distance_m.items())
        },
        "request_first_seen_s": dict(sorted(request_first_seen_s.items())),
        "request_assigned_s": dict(sorted(request_assigned_s.items())),
        "assignment_wait_s": {
            reservation_id: round(request_assigned_s[reservation_id] - first_seen, 6)
            for reservation_id, first_seen in sorted(request_first_seen_s.items())
            if reservation_id in request_assigned_s
        },
    }
    (out / "run_command.json").write_text(json.dumps(cmd, indent=2), encoding="utf-8")
    (out / "request_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    if horizon_exceeded:
        raise RuntimeError(f"Simulation exceeded horizon; diagnostics written to {out / 'request_qc.json'}")
    return {
        "tripinfo": str(tripinfo),
        "summary": str(summary),
        "emissions": str(emissions),
        "request_qc": str(out / "request_qc.json"),
    }
