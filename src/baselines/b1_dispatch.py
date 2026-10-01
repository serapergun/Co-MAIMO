"""B1: deterministic sequential nearest-feasible taxi dispatch."""

def dispatch(traci, reservations, fleet, now):
    # getTaxiReservations(0) returns all active reservations. Reservation state 1
    # denotes a newly received reservation in SUMO's taxi API.
    pending = [r for r in reservations if r.state == 1]
    idle = list(traci.vehicle.getTaxiFleet(0))
    assignments = []
    used = set()

    for reservation in sorted(pending, key=lambda r: (r.reservationTime, r.id)):
        best = None
        for vehicle_id in idle:
            if vehicle_id in used:
                continue
            edge = traci.vehicle.getRoadID(vehicle_id)
            if not edge or edge.startswith(":"):
                continue
            try:
                route = traci.simulation.findRoute(
                    edge,
                    reservation.fromEdge,
                    vType=traci.vehicle.getTypeID(vehicle_id),
                )
            except Exception:
                continue
            if not route.edges:
                continue
            candidate = (route.travelTime, vehicle_id)
            if best is None or candidate < best:
                best = candidate

        if best is not None:
            assignments.append((best[1], [reservation.id]))
            used.add(best[1])

    return assignments
