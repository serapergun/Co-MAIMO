"""Seeded safe Barcelona demand realizations for paired B4-B6 evaluation."""

import numpy as np


FRAGILE_EDGE_PREFIXES = (
    "87504601#3", "-306281847#2", "-545747621", "-545021320#3",
    "-545747622#3", "-753154730#3", "-285182795", "533269953",
    "300828583#0",
)


def safe_od_pool(net, max_pairs=96):
    edges = [
        e for e in net.getEdges()
        if not e.getID().startswith(":")
        and any(l.allows("passenger") for l in e.getLanes())
    ]
    pairs = []
    for o in edges[:3000]:
        for d in edges[-1:max(-3001, -len(edges)):-20]:
            if o.getID() == d.getID():
                continue
            route, _ = net.getOptimalPath(o, d, vClass="passenger")
            if not route:
                continue
            path = list(route)
            length = sum(e.getLength() for e in path)
            ol = [l for l in o.getLanes() if l.allows("passenger")]
            dl = [l for l in d.getLanes() if l.allows("passenger")]
            if not (3000 <= length <= 6000):
                continue
            if max((l.getLength() for l in ol), default=0) < 80:
                continue
            if max((l.getLength() for l in dl), default=0) < 80:
                continue
            if min((e.getLength() for e in path), default=0) < 25:
                continue
            if not all(any(l.allows("passenger") for l in e.getLanes()) for e in path):
                continue
            if any(
                e.getID().startswith(prefix)
                for e in path for prefix in FRAGILE_EDGE_PREFIXES
            ):
                continue
            pairs.append((o.getID(), d.getID()))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def seeded_demand(pairs, n_requests, seed, wave_size=12, wave_gap_s=1800):
    if n_requests <= 0 or wave_size <= 0:
        raise ValueError("request and wave sizes must be positive")
    if len(pairs) < wave_size:
        raise ValueError("safe OD pool smaller than one demand wave")
    rng = np.random.default_rng(int(seed))
    # Sample without replacement within each wave when possible, preserving
    # comparable demand volume while changing the OD realization by seed.
    rows = []
    for rid in range(n_requests):
        position = rid % wave_size
        if position == 0:
            current = rng.choice(len(pairs), size=wave_size, replace=False)
        pair = pairs[int(current[position])]
        rows.append({
            "request_id": f"request{rid}",
            "depart_s": 10 + (rid // wave_size) * wave_gap_s,
            "origin_edge": pair[0],
            "destination_edge": pair[1],
            "demand_seed": int(seed),
        })
    return rows
