"""Read-only, explicit-availability probes for B4-S4 TraCI candidate features.

Probes never replace missing telemetry with fabricated zero values.
"""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class Probe:
    value: float | None
    available: bool
    reason: str

def _safe_float(value):
    try:
        x=float(value)
    except (TypeError,ValueError,OverflowError):
        return None
    return x if math.isfinite(x) else None

def lane_traffic_density(traci, vehicle_id):
    """Vehicles per kilometre on current lane; not a prediction of route traffic."""
    try:
        lane=str(traci.vehicle.getLaneID(vehicle_id))
        if not lane or lane.startswith(":"):
            return Probe(None,False,"no_regular_lane")
        count=_safe_float(traci.lane.getLastStepVehicleNumber(lane))
        length=_safe_float(traci.lane.getLength(lane))
        if count is None or length is None or count<0 or length<=0:
            return Probe(None,False,"invalid_lane_telemetry")
        return Probe(1000.0*count/length,True,"observed_lane_vehicles_per_km")
    except (AttributeError,KeyError,RuntimeError,ValueError):
        return Probe(None,False,"lane_telemetry_unavailable")

def battery_soc(traci, vehicle_id):
    """Return SoC only if SUMO reports both actual and maximum battery energy."""
    try:
        actual=_safe_float(traci.vehicle.getParameter(vehicle_id,"device.battery.actualBatteryCapacity"))
        maximum=_safe_float(traci.vehicle.getParameter(vehicle_id,"device.battery.maximumBatteryCapacity"))
        if actual is None or maximum is None or maximum<=0 or actual<0 or actual>maximum:
            return Probe(None,False,"invalid_or_missing_battery_capacity")
        return Probe(actual/maximum,True,"observed_battery_soc")
    except (AttributeError,KeyError,RuntimeError,ValueError):
        return Probe(None,False,"battery_telemetry_unavailable")

def candidate_telemetry(traci, vehicle_id):
    return {"lane_density":lane_traffic_density(traci,vehicle_id),
            "battery_soc":battery_soc(traci,vehicle_id)}
