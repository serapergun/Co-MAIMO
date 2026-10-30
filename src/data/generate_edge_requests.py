"""Convert joint time-dependent district OD probabilities to edge-level requests."""
from __future__ import annotations
import numpy as np, pandas as pd

def generate_requests(od, edge_map, n_requests, seed, min_route_validator=None):
    rng=np.random.default_rng(seed)
    x=od.copy()
    if (x["weighted_n"]<0).any() or x["weighted_n"].sum()<=0:
        raise ValueError("Invalid OD weights.")
    p=x["weighted_n"].to_numpy(float); p/=p.sum()
    chosen=x.iloc[rng.choice(len(x),size=n_requests,p=p)].reset_index(drop=True)
    pools={d:g["edge_id"].astype(str).tolist() for d,g in edge_map.groupby("district")}
    rows=[]
    for k,r in chosen.iterrows():
        if r.origin not in pools or r.destination not in pools:
            raise ValueError("OD district has no mapped SUMO edge.")
        accepted=None
        for _ in range(100):
            o=rng.choice(pools[r.origin]); d=rng.choice(pools[r.destination])
            if o==d: continue
            if min_route_validator and not min_route_validator(o,d): continue
            accepted=(o,d); break
        if accepted is None:
            raise RuntimeError(f"Could not sample feasible edges for {r.origin}->{r.destination}")
        rows.append({"request_id":f"req_{k:07d}","time_bin":int(r.time_bin),
                     "origin_district":r.origin,"destination_district":r.destination,
                     "origin_edge":accepted[0],"destination_edge":accepted[1]})
    return pd.DataFrame(rows)
