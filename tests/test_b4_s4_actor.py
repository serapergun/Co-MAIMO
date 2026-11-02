import numpy as np
import pytest
from src.marl.b4_s4_actor import B4S4Actor
from src.marl.telemetry_probes import Probe

def measured(value):
    return Probe(value,True,"observed")

def missing():
    return Probe(None,False,"missing")

def test_masked_actions_never_select_infeasible():
    actor=B4S4Actor()
    probs=actor.masked_action_probs([.2,.1,.3],[True,False,True],
                                   [measured(1),measured(2),missing()],
                                   [measured(.5),measured(.6),missing()])
    assert probs[1]==0
    assert probs.sum()==pytest.approx(1.0)

def test_candidate_permutation_equivariance():
    actor=B4S4Actor()
    times=np.array([.4,.2,.7,.1])
    mask=np.array([True,False,True,True])
    density=[measured(2),missing(),measured(1),measured(0)]
    soc=[measured(.5),missing(),measured(.4),measured(.3)]
    p=actor.masked_action_probs(times,mask,density,soc)
    perm=[3,1,0,2]
    q=actor.masked_action_probs(times[perm],mask[perm],
                               [density[i] for i in perm],[soc[i] for i in perm])
    assert np.allclose(q,p[perm])

def test_missing_telemetry_is_supported():
    p=B4S4Actor().masked_action_probs([.2,.5],[True,True],[missing(),missing()],[missing(),missing()])
    assert np.isfinite(p).all() and p.sum()==pytest.approx(1)

def test_all_masked_rejected():
    with pytest.raises(ValueError):
        B4S4Actor().masked_action_probs([.2],[False],[missing()],[missing()])
