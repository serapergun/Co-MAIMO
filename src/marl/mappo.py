"""Dependency-light MAPPO policy core for the B4 benchmark."""

from dataclasses import dataclass
import numpy as np


def _softmax(logits):
    z=np.asarray(logits,dtype=float); z=z-np.max(z); e=np.exp(z)
    return e/np.sum(e)


def _masked_probs(probs, mask):
    mask=np.asarray(mask,dtype=bool)
    probs=np.asarray(probs,dtype=float)
    if probs.shape != mask.shape:
        raise ValueError("probability/mask shape mismatch")
    out=np.where(mask,probs,0.0); total=float(out.sum())
    if total <= 0:
        raise ValueError("action mask has no feasible action")
    return out/total


@dataclass
class Transition:
    observation: np.ndarray
    global_state: np.ndarray
    action: int
    old_log_prob: float
    advantage: float
    return_: float
    action_mask: np.ndarray | None = None


class LinearMAPPO:
    """Shared linear actor with centralized linear critic and PPO clipping."""

    def __init__(self,obs_dim,state_dim,action_dim,seed=0):
        if min(obs_dim,state_dim,action_dim)<=0: raise ValueError("MAPPO dimensions must be positive")
        self.obs_dim=int(obs_dim); self.state_dim=int(state_dim); self.action_dim=int(action_dim)
        rng=np.random.default_rng(seed)
        self.actor_w=rng.normal(0.0,0.01,(self.action_dim,self.obs_dim))
        self.critic_w=np.zeros(self.state_dim,dtype=float)

    def action_probs(self,observation):
        obs=np.asarray(observation,dtype=float)
        if obs.shape!=(self.obs_dim,): raise ValueError("observation shape mismatch")
        return _softmax(self.actor_w@obs)

    def masked_action_probs(self,observation,mask):
        return _masked_probs(self.action_probs(observation),mask)

    def act(self,observation,rng):
        probs=self.action_probs(observation); action=int(rng.choice(self.action_dim,p=probs))
        return action,float(np.log(max(probs[action],1e-15)))

    def value(self,global_state):
        state=np.asarray(global_state,dtype=float)
        if state.shape!=(self.state_dim,): raise ValueError("global_state shape mismatch")
        return float(self.critic_w@state)

    def update(self,transitions,actor_lr=1e-2,critic_lr=1e-2,clip_eps=0.2):
        if not transitions: raise ValueError("MAPPO update requires transitions")
        if not 0.0 < float(clip_eps) < 1.0:
            raise ValueError("clip_eps must be in (0,1)")
        raw_adv=np.asarray([float(t.advantage) for t in transitions])
        # Standardize only when dispersion exists; preserves a valid signal for
        # one-transition/same-advantage batches.
        if len(raw_adv)>1 and float(raw_adv.std())>1e-12:
            advs=(raw_adv-raw_adv.mean())/(raw_adv.std()+1e-8)
        else:
            advs=raw_adv

        actor_grad=np.zeros_like(self.actor_w); critic_grad=np.zeros_like(self.critic_w)
        policy_objective=0.0; value_loss=0.0
        for t,adv in zip(transitions,advs):
            obs=np.asarray(t.observation,dtype=float); state=np.asarray(t.global_state,dtype=float)
            base_probs=self.action_probs(obs)
            mask=np.ones(self.action_dim,dtype=bool) if t.action_mask is None else np.asarray(t.action_mask,dtype=bool)
            probs=_masked_probs(base_probs,mask)
            if not mask[t.action]: raise ValueError("recorded action is infeasible under stored mask")
            new_log_prob=float(np.log(max(probs[t.action],1e-15)))
            ratio=float(np.exp(new_log_prob-t.old_log_prob))
            clipped=float(np.clip(ratio,1.0-clip_eps,1.0+clip_eps))
            unclipped_obj=ratio*float(adv); clipped_obj=clipped*float(adv)
            # The PPO surrogate is min(ratio*A, clip(ratio)*A).
            # Differentiate only when the *strictly* unclipped branch determines
            # the objective. At a clipping plateau/boundary we choose zero
            # subgradient, avoiding an artificial push farther into saturation.
            use_unclipped=unclipped_obj < clipped_obj or abs(ratio-clipped) < 1e-12
            at_clip_boundary=(
                abs(ratio-(1.0-clip_eps)) < 1e-12
                or abs(ratio-(1.0+clip_eps)) < 1e-12
            )
            policy_objective+=min(unclipped_obj,clipped_obj)
            if use_unclipped and not at_clip_boundary:
                grad_logp=np.zeros_like(self.actor_w)
                # Gradient of masked softmax: infeasible logits have zero effect.
                feasible=np.flatnonzero(mask)
                for j in feasible:
                    grad_logp[j]=-probs[j]*obs
                grad_logp[t.action]+=obs
                actor_grad+=float(adv)*ratio*grad_logp
            prediction=self.value(state); error=prediction-float(t.return_)
            value_loss+=0.5*error*error; critic_grad+=error*state
        n=float(len(transitions))
        self.actor_w+=actor_lr*actor_grad/n; self.critic_w-=critic_lr*critic_grad/n
        return {"policy_objective":policy_objective/n,"value_loss":value_loss/n}
