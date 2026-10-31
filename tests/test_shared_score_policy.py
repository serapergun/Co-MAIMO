"""Unit tests for shared-score candidate permutation and feasibility."""
import numpy as np
import pytest
from src.marl.shared_score_policy import SharedAccessScorePolicy

def test_nearest_feasible_is_argmax():
    p=SharedAccessScorePolicy()
    x=np.array([.8,.1,.3,.05]);mask=np.array([True,False,True,True])
    assert p.select(x,mask)==3
    assert p.masked_action_probs(x,mask)[1]==0

def test_permutation_equivariance():
    p=SharedAccessScorePolicy()
    x=np.array([.7,.2,.9,.4,.1,.8,.3,.6])
    m=np.array([True,True,False,True,True,False,True,True])
    perm=np.array([7,2,0,6,4,1,5,3])
    assert np.allclose(p.masked_action_probs(x[perm],m[perm]),
                       p.masked_action_probs(x,m)[perm],atol=1e-12)

def test_equal_time_equal_probs():
    p=SharedAccessScorePolicy()
    assert np.allclose(p.masked_action_probs(np.ones(8),np.ones(8,dtype=bool)),np.ones(8)/8)

def test_invalid_inputs():
    p=SharedAccessScorePolicy()
    with pytest.raises(ValueError):p.masked_action_probs([.1,.2],[False,False])
    with pytest.raises(ValueError):SharedAccessScorePolicy(0)
