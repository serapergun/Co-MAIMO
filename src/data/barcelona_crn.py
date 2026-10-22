"""Seeded safe Barcelona demand realizations for paired B4-B6 evaluation."""

from collections import defaultdict
from math import floor

import numpy as np


FRAGILE_EDGE_PREFIXES = (
    "87504601#3", "-306281847#2", "-545747621", "-545021320#3",
    "-545747622#3", "-753154730#3", "-285182795", "533269953",
    "300828583#0",
)
_GRID_SIZE_M = 1500.0
_GRID_OFFSETS = (
    (2, 0), (-2, 0), (0, 2), (0, -2),
    (2, 1), (2, -1), (-2, 1), (-2, -1),
    (3, 0), (-3, 0), (0, 3), (0, -3),
)


def _eligible_edge(edge):
    if edge.getID().startswith(":"):
        return False
    return max(
        (lane.getLength() for lane in edge.getLanes() if lane.allows("passenger")),
        default=0.0,
    ) >= 80.0


def _edge_coordinate(edge):
    shape = edge.getShape()
    if shape:
        return shape[len(shape) // 2]
    return edge.getFromNode().getCoord()


def _route_is_safe(path):
    if not path:
        return False
    length = sum(edge.getLength() for edge in path)
    if not 3000.0 <= length <= 6000.0:
        return False
    return (
        all(any(lane.allows("passenger") for lane in edge.getLanes()) for edge in path)
        and not any(
            edge.getID().startswith(prefix)
            for edge in path
            for prefix in FRAGILE_EDGE_PREFIXES
        )
    )


def safe_od_pool(net, max_pairs=96, max_route_queries=2048):
    """Build a bounded, deterministic pool of short, passenger-safe OD pairs.

    A Cartesian scan of a city-scale network can make hundreds of thousands of
    shortest-path calls before it finds enough routes. Candidate OD edges are
    therefore selected from spatially separated grid cells. The route-query cap
    keeps preflight latency bounded without changing demand seeds or B4-B6
    pairing once the pool has been constructed.
    """
    if max_pairs <= 0 or max_route_queries <= 0:
        raise ValueError("max_pairs and max_route_queries must be positive")

    edges = [edge for edge in net.getEdges() if _eligible_edge(edge)]
    if not edges:
        return []

    coordinates = {edge.getID(): _edge_coordinate(edge) for edge in edges}
    edges.sort(key=lambda edge: (*coordinates[edge.getID()], edge.getID()))

    buckets = defaultdict(list)
    for edge in edges:
        x, y = coordinates[edge.getID()]
        buckets[(floor(x / _GRID_SIZE_M), floor(y / _GRID_SIZE_M))].append(edge)
    for bucket in buckets.values():
        bucket.sort(key=lambda edge: edge.getID())

    origin_count = min(len(edges), 768)
    origin_indices = [
        min(len(edges) - 1, (index * len(edges)) // origin_count)
        for index in range(origin_count)
    ]

    pairs = []
    seen_pairs = set()
    route_queries = 0
    for origin_position, origin_index in enumerate(origin_indices):
        origin = edges[origin_index]
        ox, oy = coordinates[origin.getID()]
        origin_cell = (floor(ox / _GRID_SIZE_M), floor(oy / _GRID_SIZE_M))

        for offset_position, (dx, dy) in enumerate(_GRID_OFFSETS):
            candidates = buckets.get((origin_cell[0] + dx, origin_cell[1] + dy), [])
            if not candidates:
                continue
            destination = candidates[
                (origin_position * 17 + offset_position * 31) % len(candidates)
            ]
            pair = (origin.getID(), destination.getID())
            if origin is destination or pair in seen_pairs:
                continue

            route_queries += 1
            route, _ = net.getOptimalPath(origin, destination, vClass="passenger")
            if _route_is_safe(list(route) if route else []):
                pairs.append(pair)
                seen_pairs.add(pair)
                if len(pairs) >= max_pairs:
                    return pairs
            if route_queries >= max_route_queries:
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
