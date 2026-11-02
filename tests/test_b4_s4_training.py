import numpy as np
import pytest
from src.marl.b4_s4_training import TrainableB4S4Actor

def transitions():
    actor=TrainableB4S4Actor()
    x=np.array([[.2,.3,.4,1,1],[.6,.1,.5,1,1],[.8,0,0,0,0]],float)
    mask=np.array([True,True,False])
    logits=x[mask]@actor.weights
    p=np.exp(logits-logits.max())
    p=p/p.sum()
    return [{"candidate_features":x,"feasible_mask":mask,"action":0,
             "old_log_prob":float(np.log(p[0])),"advantage":.5,
             "return_":-.2,"global_state":[.5,.5,.2]},
            {"candidate_features":x,"feasible_mask":mask,"action":1,
             "old_log_prob":float(np.log(p[1])),"advantage":-.5,
             "return_":-.4,"global_state":[.5,.5,.2]}]

def test_finite_masked_ppo_update():
    actor=TrainableB4S4Actor()
    before=actor.weights.copy()
    result=actor.update(transitions())
    assert result["transition_count"]==2
    assert np.isfinite(actor.weights).all()
    assert not np.allclose(before,actor.weights)

def test_infeasible_action_rejected():
    rows=transitions()
    rows[0]["action"]=2
    with pytest.raises(ValueError):TrainableB4S4Actor().update(rows)

def test_critic_shape_validation():
    with pytest.raises(ValueError):TrainableB4S4Actor().value([1,2])
