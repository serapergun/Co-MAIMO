import pandas as pd
import pytest
from src.evaluation.energy_metrics import physical_energy_metrics

def test_energy_intensity():
    df=pd.DataFrame({"routeLength":[10000,5000],
                     "energy_consumed_Wh":[2000,1000],
                     "energy_regenerated_Wh":[200,100]})
    m=physical_energy_metrics(df)
    assert m["vehicle_km"] == 15
    assert m["net_energy_kWh"] == pytest.approx(2.7)
    assert m["kWh_per_vehicle_km"] == pytest.approx(0.18)

def test_invalid_energy_fails():
    df=pd.DataFrame({"routeLength":[1000],
                     "energy_consumed_Wh":[float("nan")],
                     "energy_regenerated_Wh":[0]})
    with pytest.raises(ValueError):
        physical_energy_metrics(df)
