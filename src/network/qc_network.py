"""Structural QC for a passenger SUMO network."""
from __future__ import annotations
import json
from pathlib import Path
import sumolib

def _allows_passenger(edge):
    lanes = edge.getLanes()
    return any(lane.allows("passenger") for lane in lanes)

def inspect_network(net_file):
    net = sumolib.net.readNet(str(net_file))
    edges = [e for e in net.getEdges() if _allows_passenger(e)]
    if not edges:
        raise ValueError("No passenger-accessible edges.")
    edge_ids = {e.getID() for e in edges}
    adjacency = {eid: set() for eid in edge_ids}
    reverse = {eid: set() for eid in edge_ids}
    for e in edges:
        for out_edge in e.getOutgoing().keys():
            oid = out_edge.getID()
            if oid in edge_ids:
                adjacency[e.getID()].add(oid)
                reverse[oid].add(e.getID())
    isolated = [eid for eid in edge_ids if not adjacency[eid] and not reverse[eid]]

    # Tarjan SCC over passenger-accessible directed edges.  This avoids adding
    # another graph dependency and makes the manuscript's >=0.95 gate executable.
    index = 0
    stack = []
    on_stack = set()
    indices = {}
    lowlink = {}
    components = []

    def strongconnect(v):
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)
        for w in adjacency[v]:
            if w not in indices:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], indices[w])
        if lowlink[v] == indices[v]:
            comp = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                comp.append(w)
                if w == v:
                    break
            components.append(comp)

    for eid in edge_ids:
        if eid not in indices:
            strongconnect(eid)

    largest_scc_edges = max((len(x) for x in components), default=0)
    largest_scc_share = largest_scc_edges / len(edges)
    scc_threshold = 0.95
    return {
        "passenger_edges": len(edges),
        "isolated_passenger_edges": len(isolated),
        "isolated_edge_ids": sorted(isolated),
        "total_length_m": float(sum(e.getLength() for e in edges)),
        "strongly_connected_components": len(components),
        "largest_scc_edges": largest_scc_edges,
        "largest_scc_share": largest_scc_share,
        "scc_acceptance_threshold": scc_threshold,
        "scc_gate_pass": largest_scc_share >= scc_threshold,
    }

def write_report(net_file, output_json):
    r = inspect_network(net_file)
    Path(output_json).write_text(json.dumps(r, indent=2), encoding="utf-8")
    return r
