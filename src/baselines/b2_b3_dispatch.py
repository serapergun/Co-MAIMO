"""B2 static two-zone dispatch and B3 backlog-threshold batch dispatch.

Both are deterministic, non-learning comparators. No test-seed tuning.
"""
import hashlib
from src.baselines.b1_dispatch import dispatch as b1_dispatch

def _travel(traci, vehicle_id, reservation):
    edge=traci.vehicle.getRoadID(vehicle_id)
    if not edge or edge.startswith(":"):
        return None
    try:
        route=traci.simulation.findRoute(edge,reservation.fromEdge,vType=traci.vehicle.getTypeID(vehicle_id))
    except Exception:
        return None
    return float(route.travelTime) if route.edges else None

def b2_static_zone_dispatch(traci,reservations,fleet,now):
    """Static two-zone fleet split: taxi0..3 vs taxi4..7.
    A request's origin is mapped to a fixed zone by stable SHA256 hash.
    A cross-zone fallback is used only if no zone taxi is idle/feasible.
    """
    idle=sorted(traci.vehicle.getTaxiFleet(0))
    used=set()
    assignments=[]
    for r in sorted(reservations,key=lambda x:(x.reservationTime,x.id)):
        zone=int.from_bytes(hashlib.sha256(r.fromEdge.encode()).digest()[:4],"big")%2
        ranked=[]
        for v in idle:
            if v in used: continue
            cost=_travel(traci,v,r)
            if cost is None: continue
            try:
                taxi_index=int(v.removeprefix("taxi"))
            except ValueError:
                taxi_index=0
            home_zone=0 if taxi_index<4 else 1
            ranked.append((home_zone!=zone,cost,v))
        if ranked:
            _,_,v=min(ranked)
            assignments.append((v,[r.id]))
            used.add(v)
    return assignments

def b3_threshold_dispatch(traci,reservations,fleet,now):
    """Demand-triggered batch matching: global shortest feasible pair first
    when >=2 reservations are pending; otherwise sequential B1 dispatch.
    """
    if len(reservations)<2:
        return b1_dispatch(traci,reservations,fleet,now)
    idle=sorted(traci.vehicle.getTaxiFleet(0))
    pending={r.id:r for r in reservations}
    assigned=[]
    while pending and idle:
        candidates=[]
        for r in pending.values():
            for v in idle:
                cost=_travel(traci,v,r)
                if cost is not None:
                    candidates.append((cost,r.reservationTime,r.id,v))
        if not candidates: break
        _,_,rid,v=min(candidates)
        assigned.append((v,[rid]))
        idle.remove(v)
        del pending[rid]
    return assigned
