"""Trainable taxi-permutation-equivariant actor with masked PPO updates.

A deliberately small architecture baseline: each taxi receives the same
linear access-time score; the critic is centralized and linear. Unlike the
fixed SharedAccessScorePolicy, the actor coefficient is learned from rewards.
"""
import numpy as np
from src.marl.mappo import Transition

class SharedScoreMAPPO:
    def __init__(self,action_dim,state_dim=3,seed=0):
        if action_dim<=0 or state_dim<=0:raise ValueError("invalid dimensions")
        self.action_dim=int(action_dim);self.obs_dim=int(action_dim);self.state_dim=int(state_dim)
        self.actor_w=np.array([-0.01],dtype=float)
        self.critic_w=np.zeros(self.state_dim,dtype=float)
    def masked_action_probs(self,observation,mask):
        obs=np.asarray(observation,dtype=float);m=np.asarray(mask,dtype=bool)
        if obs.shape!=(self.action_dim,) or m.shape!=obs.shape or not m.any():raise ValueError("invalid mask or observation")
        if not np.all(np.isfinite(obs[m])):raise ValueError("nonfinite feasible observation")
        logits=self.actor_w[0]*obs[m];logits-=np.max(logits)
        e=np.exp(logits);p=np.zeros(self.action_dim,dtype=float);p[m]=e/e.sum()
        return p
    def value(self,state):
        x=np.asarray(state,dtype=float)
        if x.shape!=(self.state_dim,):raise ValueError("invalid critic state")
        return float(self.critic_w@x)
    def update(self,transitions,actor_lr=1e-2,critic_lr=1e-2,clip_eps=.2):
        if not transitions or not 0<clip_eps<1:raise ValueError("invalid PPO batch/clip")
        adv=np.array([t.advantage for t in transitions],float)
        if len(adv)>1 and adv.std()>1e-12:adv=(adv-adv.mean())/(adv.std()+1e-8)
        grad=0.;critic_grad=np.zeros_like(self.critic_w);objective=0.;loss=0.
        for t,a in zip(transitions,adv):
            x=np.asarray(t.observation,dtype=float)
            m=np.ones(self.action_dim,dtype=bool) if t.action_mask is None else np.asarray(t.action_mask,dtype=bool)
            p=self.masked_action_probs(x,m)
            if not m[t.action]:raise ValueError("infeasible selected action")
            ratio=float(np.exp(np.log(max(p[t.action],1e-15))-t.old_log_prob))
            clipped=float(np.clip(ratio,1-clip_eps,1+clip_eps))
            objective+=min(ratio*a,clipped*a)
            # No actor gradient on PPO clipping plateaus.
            if (ratio*a <= clipped*a and abs(ratio-clipped)<1e-12) or (ratio*a < clipped*a):
                grad+=a*ratio*(x[t.action]-float(p@x))
            state=np.asarray(t.global_state,dtype=float)
            err=self.value(state)-t.return_
            loss+=.5*err*err;critic_grad+=err*state
        n=len(transitions)
        self.actor_w[0]+=actor_lr*grad/n
        self.critic_w-=critic_lr*critic_grad/n
        return {"policy_objective":float(objective/n),"value_loss":float(loss/n)}
