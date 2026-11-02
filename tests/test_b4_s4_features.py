import numpy as np
import pytest
from src.marl.b4_s4_features import encode_candidate
from src.marl.telemetry_probes import Probe

def test_valid_observed_features():
    result=encode_candidate(.2,Probe(3.0,True,"observed"),Probe(.45,True,"observed"))
    assert result.values.shape==(5,)
    assert result.values[0]==pytest.approx(.2)
    assert result.values[2]==pytest.approx(.45)
    assert np.array_equal(result.availability,[1,1])

def test_missing_values_have_separate_indicators():
    result=encode_candidate(.2,Probe(None,False,"missing"),Probe(None,False,"missing"))
    assert np.array_equal(result.values,[.2,0,0,0,0])
    assert np.array_equal(result.availability,[0,0])

@pytest.mark.parametrize("density,soc",[(Probe(-1,True,"bad"),Probe(.5,True,"ok")),(Probe(0,True,"ok"),Probe(1.2,True,"bad"))])
def test_invalid_observed_values_rejected(density,soc):
    with pytest.raises(ValueError):encode_candidate(.2,density,soc)

def test_invalid_access_rejected():
    with pytest.raises(ValueError):encode_candidate(float("nan"),Probe(None,False,"missing"),Probe(None,False,"missing"))
