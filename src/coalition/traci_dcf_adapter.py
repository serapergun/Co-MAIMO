"""B5 dispatcher: frozen MAPPO decisions plus route-grounded DCF/CCE swaps."""

import numpy as np

from src.coalition.dcf import CoalitionProposal, evaluate_merge
from src.marl.traci_adapter import MAPPOTraciDispatcher, _route_time


class DCFMAPPOTraciDispatcher(MAPPOTraciDispatcher):
    """Apply individually-rational pairwise coalition swaps to MAPPO assignments."""

    def __init__(self, policy, seed=0, coordination_cost_s=0.0):
        super().__init__(policy, seed=seed)
        if coordination_cost_s < 0:
            raise ValueError("coordination_cost_s must be non-negative")
        self.coordination_cost_s = float(coordination_cost_s)
        self.coalition_events = []

    def __call__(self, traci, reservations, fleet, now):
        assignments = super().__call__(traci, reservations, fleet, now)
        if len(assignments) < 2:
            return assignments

        reservation_by_id = {str(r.id): r for r in reservations}
        revised = list(assignments)

        # Deterministic disjoint adjacent pairs avoid order-dependent repeated
        # swapping of the same taxi within one simulation step.
        for i in range(0, len(revised) - 1, 2):
            taxi_a, ids_a = revised[i]
            taxi_b, ids_b = revised[i + 1]
            if len(ids_a) != 1 or len(ids_b) != 1:
                continue
            ra = reservation_by_id.get(str(ids_a[0]))
            rb = reservation_by_id.get(str(ids_b[0]))
            if ra is None or rb is None:
                continue

            aa = _route_time(traci, taxi_a, ra)
            bb = _route_time(traci, taxi_b, rb)
            ab = _route_time(traci, taxi_a, rb)
            ba = _route_time(traci, taxi_b, ra)
            if any(x is None for x in (aa, bb, ab, ba)):
                continue

            half_coord = self.coordination_cost_s / 2.0
            proposal = CoalitionProposal(
                members=(taxi_a, taxi_b),
                independent_cost=float(aa + bb),
                cooperative_cost=float(ab + ba),
                coordination_cost=self.coordination_cost_s,
                member_independent_costs={taxi_a: float(aa), taxi_b: float(bb)},
                member_cooperative_costs={taxi_a: float(ab), taxi_b: float(ba)},
                member_coordination_costs={taxi_a: half_coord, taxi_b: half_coord},
            )
            decision = evaluate_merge(proposal)
            self.coalition_events.append({
                "time": float(now),
                "members": [taxi_a, taxi_b],
                "reservation_ids": [str(ra.id), str(rb.id)],
                "accepted": bool(decision.accepted),
                "independent_cost_s": float(aa + bb),
                "cooperative_cost_s": float(ab + ba),
                "coordination_cost_s": float(self.coordination_cost_s),
                "system_break_even_overhead_s": float((aa + bb) - (ab + ba)),
                "member_break_even_overhead_s": {
                    taxi_a: float(2.0 * (aa - ab)),
                    taxi_b: float(2.0 * (bb - ba)),
                },
                "coalition_gain_s": float(decision.coalition_gain),
                "member_gains_s": decision.member_gains,
                "reason": decision.reason,
            })
            if decision.accepted:
                revised[i] = (taxi_a, [rb.id])
                revised[i + 1] = (taxi_b, [ra.id])

                # MAPPO sampled the taxis, but DCF changed which reservation
                # each sampled taxi actually serves. Training must therefore
                # join lifecycle reward to the applied taxi-reservation pair,
                # while preserving the proposal reservation for audit.
                by_vehicle = {
                    str(d["vehicle_id"]): d
                    for d in self.decisions
                    if float(d["time"]) == float(now)
                }
                da = by_vehicle.get(str(taxi_a))
                db = by_vehicle.get(str(taxi_b))
                if da is None or db is None:
                    raise RuntimeError("accepted DCF swap lacks MAPPO decision record")
                da["proposed_reservation_id"] = str(da["reservation_id"])
                db["proposed_reservation_id"] = str(db["reservation_id"])
                da["reservation_id"] = str(rb.id)
                db["reservation_id"] = str(ra.id)
                da["dcf_swap_applied"] = True
                db["dcf_swap_applied"] = True

        return revised
