"""B4-S3 permutation-equivariant multi-feature masked PPO actor.

The actor is shared across taxis: z_i = w @ x_i + b, with x_i=(normalized
access time, normalized access-time gap from best feasible taxi). The second
feature is candidate-relative and does not require new SUMO sensors. An
independent centralized linear critic uses the existing three state features.
"""
import numpy as np
from src.marl.shared_score_mappo import SharedScoreMAPPO

class MultiFeatureMAPPO(SharedScoreMAPPO):
    def __init__(self,action_dim,state_dim=3,seed=0):
        super().__init__(action_dim,state_dim,seed)
        self.actor_w=np.array([-2.0,-1.0],dtype=float)
        self.architecture="shared_two_feature_ppo_v1"

    def features(self,observation,mask):
        obs=np.asarray(observation,dtype=float)
        feasible=np.asarray(mask,dtype=bool)
        if obs.shape!=(self.action_dim,) or feasible.shape!=obs.shape or not feasible.any():
            raise ValueError("invalid candidate observation or mask")
        if not np.all(np.isfinite(obs[feasible])):
            raise ValueError("nonfinite candidate features")
        best=float(obs[feasible].min())
        return np.stack((obs,obs-best),axis=-1)

    def masked_action_probs(self,observation,mask):
        feasible=np.asarray(mask,dtype=bool)
        x=self.features(observation,feasible)
        logits=x[feasible]@self.actor_w
        logits=logits-np.max(logits)
        e=np.exp(logits)
        p=np.zeros(self.action_dim,dtype=float)
        p[feasible]=e/e.sum()
        return p

    def update(self,transitions,actor_lr=.5,critic_lr=.01,clip_eps=.2):
        if not transitions or not 0<clip_eps<1:
            raise ValueError("invalid PPO batch or clip")
        raw=np.asarray([t.advantage for t in transitions],dtype=float)
        adv=(raw-raw.mean())/(raw.std()+1e-8) if len(raw)>1 and raw.std()>1e-12 else raw
        actor_grad=np.zeros_like(self.actor_w)
        critic_grad=np.zeros_like(self.critic_w)
        objective=0.;loss=0.
        for t,a in zip(transitions,adv):
            mask=np.ones(self.action_dim,dtype=bool) if t.action_mask is None else np.asarray(t.action_mask,dtype=bool)
            x=self.features(t.observation,mask)
            p=self.masked_action_probs(t.observation,mask)
            if not mask[t.action]:raise ValueError("infeasible recorded action")
            ratio=float(np.exp(np.clip(np.log(max(p[t.action],1e-15))-t.old_log_prob,-50,50)))
            clipped=float(np.clip(ratio,1-clip_eps,1+clip_eps))
            objective+=min(ratio*a,clipped*a)
            if (ratio*a<clipped*a or abs(ratio-clipped)<1e-12) and not (
                abs(ratio-(1-clip_eps))<1e-12 or abs(ratio-(1+clip_eps))<1e-12):
                actor_grad+=a*ratio*(x[t.action]-(p[:,None]*x).sum(axis=0))
            state=np.asarray(t.global_state,dtype=float)
            err=self.value(state)-float(t.return_)
            loss+=.5*err*err
            critic_grad+=err*state
        n=len(transitions)
        self.actor_w+=actor_lr*actor_grad/n
        self.critic_w-=critic_lr*critic_grad/n
        return {"policy_objective":float(objective/n),"value_loss":float(loss/n)}
