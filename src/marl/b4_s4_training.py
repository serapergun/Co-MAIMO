"""PPO training adapter for observed B4-S4 candidate feature matrices.

Preserves masked actions, frozen old log-probabilities and shared candidate weights.
"""
import numpy as np
from src.marl.b4_s4_actor import B4S4Actor

class TrainableB4S4Actor(B4S4Actor):
    def __init__(self, weights=None, state_dim=3):
        super().__init__(weights)
        self.state_dim=state_dim
        self.critic_w=np.zeros(state_dim,dtype=float)

    def value(self,state):
        x=np.asarray(state,dtype=float)
        if x.shape!=(self.state_dim,) or not np.isfinite(x).all():
            raise ValueError("invalid global state")
        return float(self.critic_w@x)

    def update(self,transitions,actor_lr=.2,critic_lr=.01,clip_eps=.2):
        if not transitions or not 0<clip_eps<1:
            raise ValueError("invalid PPO batch")
        advantages=np.asarray([t["advantage"] for t in transitions],dtype=float)
        if not np.isfinite(advantages).all():raise ValueError("nonfinite advantages")
        if len(advantages)>1 and advantages.std()>1e-12:
            advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
        actor_gradient=np.zeros(5,dtype=float)
        critic_gradient=np.zeros(self.state_dim,dtype=float)
        objective=0.0
        for t,adv in zip(transitions,advantages):
            features=np.asarray(t["candidate_features"],dtype=float)
            mask=np.asarray(t["feasible_mask"],dtype=bool)
            action=int(t["action"])
            state=np.asarray(t["global_state"],dtype=float)
            if features.ndim!=2 or features.shape[1]!=5 or mask.shape!=(len(features),) or not mask.any() or not mask[action]:
                raise ValueError("invalid masked training action")
            if not np.isfinite(features[mask]).all():raise ValueError("nonfinite training features")
            logits=features[mask]@self.weights
            logits-=logits.max()
            e=np.exp(logits)
            probs=np.zeros(len(features))
            probs[mask]=e/e.sum()
            old_log_prob=float(t["old_log_prob"])
            ratio=float(np.exp(np.clip(np.log(max(probs[action],1e-15))-old_log_prob,-50,50)))
            clipped=float(np.clip(ratio,1-clip_eps,1+clip_eps))
            objective+=min(ratio*adv,clipped*adv)
            if ratio*adv<=clipped*adv+1e-12 and 1-clip_eps<ratio<1+clip_eps:
                actor_gradient+=adv*ratio*(features[action]-(probs[:,None]*features).sum(axis=0))
            critic_gradient+=(self.value(state)-float(t["return_"]))*state
        n=len(transitions)
        self.weights+=actor_lr*actor_gradient/n
        self.critic_w-=critic_lr*critic_gradient/n
        if not np.isfinite(self.weights).all() or not np.isfinite(self.critic_w).all():
            raise FloatingPointError("nonfinite PPO parameters")
        return {"policy_objective":float(objective/n),"transition_count":n}
