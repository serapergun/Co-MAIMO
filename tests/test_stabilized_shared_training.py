"""Sanity checks for stabilized return units and actor initialization."""
import numpy as np
from src.marl.shared_score_mappo import SharedScoreMAPPO
from src.marl.stabilized_training import REWARD_SCALE_S

def test_scaled_critic_units():
    assert REWARD_SCALE_S==1000.
    policy=SharedScoreMAPPO(8,state_dim=3)
    assert policy.value(np.array([1.,.5,.2]))==0.

def test_nearest_prior_permutation_and_mask():
    policy=SharedScoreMAPPO(4,state_dim=3)
    policy.actor_w[0]=-2.
    x=np.array([.7,.2,.9,.4])
    mask=np.array([True,True,False,True])
    p=policy.masked_action_probs(x,mask)
    assert int(np.argmax(p))==1
    assert p[2]==0
    perm=np.array([3,0,2,1])
    assert np.allclose(policy.masked_action_probs(x[perm],mask[perm]),p[perm])
