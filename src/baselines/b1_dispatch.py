"""B1: deterministic sequential nearest-feasible taxi dispatch."""

def dispatch(traci, reservations, fleet, now):
    """Assign each newly surfaced reservation to the nearest feasible idle taxi.

    The runner supplies getTaxiReservations(onlyNew=1), so this function does
    not depend on undocumented numeric reservation-state constants.
    """
    idle = list(traci.vehicle.getTaxiFleet(0))
    assignments = []
    used = set()

    for reservation in sorted(reservations, key=lambda r: (r.reservationTime, r.id)):
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
