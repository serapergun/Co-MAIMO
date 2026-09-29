"""Structural QC for a passenger SUMO network."""
from __future__ import annotations
import json
from pathlib import Path
import sumolib

def inspect_network(net_file):
    net=sumolib.net.readNet(str(net_file))
    edges=[e for e in net.getEdges() if "passenger" in e.getAllowed()]
    if not edges:
        raise ValueError("No passenger-accessible edges.")
    edge_ids={e.getID() for e in edges}
    adjacency={eid:set() for eid in edge_ids}
    reverse={eid:set() for eid in edge_ids}
    for e in edges:
        for c in e.getOutgoing().keys():
            if c.getID() in edge_ids:
                adjacency[e.getID()].add(c.getID()); reverse[c.getID()].add(e.getID())
    isolated=[e for e in edge_ids if not adjacency[e] and not reverse[e]]
    return {
        "passenger_edges":len(edges),
        "isolated_passenger_edges":len(isolated),
        "isolated_edge_ids":sorted(isolated),
        "total_length_m":sum(e.getLength() for e in edges),
    }

def write_report(net_file, output_json):
    r=inspect_network(net_file)
    Path(output_json).write_text(json.dumps(r,indent=2),encoding="utf-8")
    return r
