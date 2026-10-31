"""Controlled gradient and symmetry checks for trainable shared-score actor."""
import numpy as np
from src.marl.shared_score_mappo import SharedScoreMAPPO
from src.marl.mappo import Transition

def test_permutation_equivariance():
    p=SharedScoreMAPPO(8)
    x=np.array([.8,.1,.5,.4,.3,.7,.2,.9]);m=np.array([1,1,0,1,1,0,1,1],bool)
    perm=np.array([3,0,6,4,7,2,1,5])
    assert np.allclose(p.masked_action_probs(x[perm],m[perm]),p.masked_action_probs(x,m)[perm])

def test_mask_and_normalization():
    p=SharedScoreMAPPO(4)
    prob=p.masked_action_probs([.8,.1,.3,.6],[1,0,1,1])
    assert prob[1]==0 and np.isclose(prob.sum(),1)

def test_ppo_gradient_improves_chosen_action():
    p=SharedScoreMAPPO(3)
    x=np.array([.1,.5,.9]);m=np.ones(3,bool)
    old=p.masked_action_probs(x,m)
    t=Transition(observation=x,global_state=np.array([1.,0.,0.]),action=0,
                 old_log_prob=float(np.log(old[0])),advantage=1.,return_=1.,action_mask=m)
    p.update([t],actor_lr=.5)
    new=p.masked_action_probs(x,m)
    assert new[0]>old[0]

def test_gradient_finite_difference():
    p=SharedScoreMAPPO(3)
    x=np.array([.1,.5,.9]);m=np.ones(3,bool)
    w=float(p.actor_w[0]);eps=1e-6
    def logp(weight):
        p.actor_w[0]=weight
        return np.log(p.masked_action_probs(x,m)[0])
    numerical=(logp(w+eps)-logp(w-eps))/(2*eps)
    p.actor_w[0]=w
    analytic=x[0]-float(p.masked_action_probs(x,m)@x)
    assert np.isclose(numerical,analytic,atol=1e-7)
