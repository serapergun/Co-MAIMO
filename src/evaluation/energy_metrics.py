"""Physical SUMO/MMPEVEM energy metrics.

No proxy or arbitrary-unit values are accepted here. Inputs must come from
SUMO tripinfo battery output produced by an executed simulation.
"""
from __future__ import annotations
import math
import pandas as pd

REQUIRED = ("routeLength", "energy_consumed_Wh", "energy_regenerated_Wh")

def physical_energy_metrics(df: pd.DataFrame) -> dict:
    x=df.copy()
    missing=[c for c in REQUIRED if c not in x.columns]
    if missing:
        raise ValueError(f"Missing physical-energy fields: {missing}")
    for c in REQUIRED:
        x[c]=pd.to_numeric(x[c], errors="coerce")
    if x[list(REQUIRED)].isna().any().any():
        raise ValueError("NaN/non-numeric MMPEVEM energy or distance record.")
    if (x["routeLength"] < 0).any():
        raise ValueError("Negative route length.")
    if (x["energy_consumed_Wh"] < 0).any() or (x["energy_regenerated_Wh"] < 0).any():
        raise ValueError("Negative cumulative battery-energy field.")
    distance_km=float(x["routeLength"].sum()/1000.0)
    if not math.isfinite(distance_km) or distance_km <= 0:
        raise ValueError("Non-positive fleet vehicle-km.")
    consumed_kwh=float(x["energy_consumed_Wh"].sum()/1000.0)
    regenerated_kwh=float(x["energy_regenerated_Wh"].sum()/1000.0)
    net_kwh=consumed_kwh-regenerated_kwh
    if not math.isfinite(net_kwh) or net_kwh < 0:
        raise ValueError("Invalid net electrical energy.")
    return {
        "vehicle_km": distance_km,
        "gross_energy_kWh": consumed_kwh,
        "regenerated_energy_kWh": regenerated_kwh,
        "net_energy_kWh": net_kwh,
        "kWh_per_vehicle_km": net_kwh/distance_km,
        "regeneration_fraction": regenerated_kwh/consumed_kwh if consumed_kwh > 0 else 0.0,
    }
