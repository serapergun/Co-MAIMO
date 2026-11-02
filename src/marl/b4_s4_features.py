"""B4-S4 observed-feature encoding with explicit missingness indicators.

No imputation is presented as an observed physical quantity.
"""
from dataclasses import dataclass
import numpy as np
from src.marl.telemetry_probes import Probe

@dataclass(frozen=True)
class EncodedCandidate:
    values: np.ndarray
    availability: np.ndarray

def encode_candidate(normalized_access_time, edge_density:Probe, soc:Probe, density_scale=10.0):
    access=float(normalized_access_time)
    if not np.isfinite(access) or access<0:
        raise ValueError("access time must be finite and nonnegative")
    if density_scale<=0 or not np.isfinite(density_scale):
        raise ValueError("density scale must be positive")
    present=np.array([bool(edge_density.available),bool(soc.available)],dtype=np.float64)
    density=0.0
    battery=0.0
    if present[0]:
        density=float(edge_density.value)
        if not np.isfinite(density) or density<0:raise ValueError("invalid observed edge density")
        density=np.log1p(density)/np.log1p(density_scale)
    if present[1]:
        battery=float(soc.value)
        if not np.isfinite(battery) or not 0<=battery<=1:raise ValueError("invalid observed SoC")
    values=np.array([access,density,battery,*present],dtype=np.float64)
    return EncodedCandidate(values,present)
