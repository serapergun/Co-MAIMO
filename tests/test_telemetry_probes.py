"""Unit tests for explicit availability of B4-S4 SUMO telemetry."""
from types import SimpleNamespace
import pytest
from src.marl.telemetry_probes import lane_traffic_density,battery_soc

def traci(lane="lane0",count=3,length=150,actual="40",maximum="50"):
    def param(_vehicle,key):
        return actual if key.endswith("actualBatteryCapacity") else maximum
    return SimpleNamespace(
        vehicle=SimpleNamespace(getLaneID=lambda _:lane,getParameter=param),
        lane=SimpleNamespace(getLastStepVehicleNumber=lambda _:count,getLength=lambda _:length))

def test_lane_density():
    p=lane_traffic_density(traci(),"taxi0")
    assert p.available and p.value==pytest.approx(20)

def test_internal_lane_missing():
    p=lane_traffic_density(traci(lane=":internal"),"taxi0")
    assert not p.available and p.value is None

def test_battery_soc():
    p=battery_soc(traci(),"taxi0")
    assert p.available and p.value==pytest.approx(.8)

@pytest.mark.parametrize("actual,maximum",[("","50"),("40","0"),("60","50"),("nan","50")])
def test_invalid_battery_is_missing(actual,maximum):
    p=battery_soc(traci(actual=actual,maximum=maximum),"taxi0")
    assert not p.available and p.value is None
