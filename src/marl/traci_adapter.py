"""TraCI adapter for the B4 MAPPO dispatch policy.

The adapter exposes one decision per reservation. Candidate actions are only
route-feasible idle taxis. A fixed action space is maintained by mapping the
current fleet order to policy actions; infeasible actions are masked before
sampling.
"""

import numpy as np


def _route_time(traci, vehicle_id, reservation):
    edge = traci.vehicle.getRoadID(vehicle_id)
    if not edge or edge.startswith(":"):
        return None
    try:
        route = traci.simulation.findRoute(
            edge,
            reservation.fromEdge,
            vType=traci.vehicle.getTypeID(vehicle_id),
        )
    except Exception:
        return None
    if not route.edges:
        return None
    return float(route.travelTime)


def reservation_features(traci, reservation, fleet):
    """Return per-taxi route-time features and feasibility mask."""
    times = []
    feasible = []
    for vehicle_id in fleet:
        travel_time = _route_time(traci, vehicle_id, reservation)
        feasible.append(travel_time is not None)
        times.append(0.0 if travel_time is None else travel_time)
    finite = [t for t, ok in zip(times, feasible) if ok]
    scale = max(finite, default=1.0)
    normalized = [
        (t / scale) if ok else 1.0
        for t, ok in zip(times, feasible)
    ]
    return np.asarray(normalized, dtype=float), np.asarray(feasible, dtype=bool)


def masked_action(policy, observation, mask, rng):
    """Sample from the policy after removing infeasible taxi actions."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return None, None
    masked = policy.masked_action_probs(observation, mask)
    action = int(rng.choice(len(masked), p=masked))
    return action, float(np.log(max(masked[action], 1e-15)))


class MAPPOTraciDispatcher:
    """Callable dispatch adapter compatible with simulation.traci_runner.run."""

    def __init__(self, policy, seed=0):
        self.policy = policy
        self.rng = np.random.default_rng(seed)
        self.decisions = []

    def __call__(self, traci, reservations, fleet, now):
        idle = tuple(sorted(traci.vehicle.getTaxiFleet(0)))
        fleet = tuple(sorted(fleet))
        if len(fleet) != self.policy.action_dim:
            raise ValueError(
                f"MAPPO action_dim={self.policy.action_dim} but fleet has {len(fleet)} taxis"
            )
        idle_set = set(idle)
        used = set()
        assignments = []

        for reservation in sorted(reservations, key=lambda r: (r.reservationTime, r.id)):
            obs, route_mask = reservation_features(traci, reservation, fleet)
            mask = np.asarray([
                bool(route_mask[i] and vehicle_id in idle_set and vehicle_id not in used)
                for i, vehicle_id in enumerate(fleet)
            ])
            action, log_prob = masked_action(self.policy, obs, mask, self.rng)
            if action is None:
                continue
            vehicle_id = fleet[action]
            assignments.append((vehicle_id, [reservation.id]))
            used.add(vehicle_id)
            self.decisions.append({
                "time": float(now),
                "reservation_id": str(reservation.id),
                "vehicle_id": vehicle_id,
                "action": int(action),
                "log_prob": float(log_prob),
                "observation": obs.tolist(),
                "selected_access_time_s": float(_route_time(traci, vehicle_id, reservation)),

                "feasible_mask": mask.astype(int).tolist(),
            })
        return assignments
