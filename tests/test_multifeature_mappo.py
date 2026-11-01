import numpy as np
import pytest
from src.marl.multifeature_mappo import MultiFeatureMAPPO
from src.marl.mappo import Transition

def test_mask_and_probabilities():
    p=MultiFeatureMAPPO(4)
    probs=p.masked_action_probs([.2,.8,.4,1.],[1,0,1,0])
    assert probs[1]==probs[3]==0
    assert np.isclose(probs.sum(),1.)
    assert probs[0]>probs[2]

def test_permutation_equivariance():
    p=MultiFeatureMAPPO(4)
    obs=np.array([.2,.9,.5,.7])
    mask=np.array([True,False,True,True])
    order=np.array([2,0,3,1])
    a=p.masked_action_probs(obs,mask)
    b=p.masked_action_probs(obs[order],mask[order])
    assert np.allclose(a[order],b)

def test_nonlinear_feature_not_constant_shift():
    p=MultiFeatureMAPPO(3)
    x=p.features([.2,.5,.9],[1,1,1])
    assert not np.allclose(x[:,1]-x[:,0],(x[:,1]-x[:,0])[0])

def test_ppo_update_finite_and_masked():
    p=MultiFeatureMAPPO(3)
    obs=np.array([.2,.5,.9]);mask=np.array([True,True,False])
    old=np.log(p.masked_action_probs(obs,mask)[0])
    t=Transition(observation=obs,global_state=np.array([1.,.5,.2]),
                 action=0,old_log_prob=float(old),advantage=1.,
                 return_=0.1,action_mask=mask)
    metrics=p.update([t])
    assert all(np.isfinite(v) for v in metrics.values())
    assert np.all(np.isfinite(p.actor_w))
    assert np.all(np.isfinite(p.critic_w))
