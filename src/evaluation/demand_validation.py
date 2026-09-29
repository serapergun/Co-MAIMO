"""Validation of generated demand against EMEF-derived calibration targets."""
from __future__ import annotations
import numpy as np
import pandas as pd

def _norm(s):
    s=np.asarray(s,dtype=float)
    z=s.sum()
    if z<=0: raise ValueError("Distribution has non-positive mass.")
    return s/z

def distribution_mae(reference, generated):
    a,b=_norm(reference),_norm(generated)
    if a.shape!=b.shape: raise ValueError("Shape mismatch.")
    return float(np.mean(np.abs(a-b)))

def vector_correlation(reference, generated):
    a,b=np.asarray(reference,float),np.asarray(generated,float)
    if a.shape!=b.shape: raise ValueError("Shape mismatch.")
    if np.std(a)==0 or np.std(b)==0: return float("nan")
    return float(np.corrcoef(a,b)[0,1])

def validate_od(reference_od, generated_requests):
    ref=(reference_od.groupby(["origin","destination"])["weighted_n"].sum())
    gen=(generated_requests.groupby(["origin_district","destination_district"]).size())
    idx=ref.index.union(gen.index)
    a=ref.reindex(idx,fill_value=0).to_numpy(float)
    b=gen.reindex(idx,fill_value=0).to_numpy(float)
    return {"od_distribution_mae":distribution_mae(a,b),
            "od_vector_correlation":vector_correlation(a,b)}

def validate_temporal(reference_od, generated_requests):
    ref=reference_od.groupby("time_bin")["weighted_n"].sum()
    gen=generated_requests.groupby("time_bin").size()
    idx=ref.index.union(gen.index)
    return {"temporal_distribution_mae":
            distribution_mae(ref.reindex(idx,fill_value=0),gen.reindex(idx,fill_value=0))}
