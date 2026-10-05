"""Dependency-light MAPPO policy core for the B4 benchmark.

The implementation provides a real stochastic actor/centralized-critic update
without introducing a deep-learning dependency.  It is intentionally small so
the policy mathematics can be unit-tested before coupling it to TraCI.  SUMO
integration and experiment results must not be claimed until the adapter and
training loop are executed.
"""

from dataclasses import dataclass
import numpy as np


def _softmax(logits):
    z = np.asarray(logits, dtype=float)
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


@dataclass
class Transition:
    observation: np.ndarray
    global_state: np.ndarray
    action: int
    old_log_prob: float
    advantage: float
    return_: float


class LinearMAPPO:
    """Shared linear actor with centralized linear critic and PPO clipping."""

    def __init__(self, obs_dim, state_dim, action_dim, seed=0):
        if min(obs_dim, state_dim, action_dim) <= 0:
            raise ValueError("MAPPO dimensions must be positive")
        self.obs_dim = int(obs_dim)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        rng = np.random.default_rng(seed)
        self.actor_w = rng.normal(0.0, 0.01, (self.action_dim, self.obs_dim))
        self.critic_w = np.zeros(self.state_dim, dtype=float)

    def action_probs(self, observation):
        obs = np.asarray(observation, dtype=float)
        if obs.shape != (self.obs_dim,):
            raise ValueError("observation shape mismatch")
        return _softmax(self.actor_w @ obs)

    def act(self, observation, rng):
        probs = self.action_probs(observation)
        action = int(rng.choice(self.action_dim, p=probs))
        return action, float(np.log(max(probs[action], 1e-15)))

    def value(self, global_state):
        state = np.asarray(global_state, dtype=float)
        if state.shape != (self.state_dim,):
            raise ValueError("global_state shape mismatch")
        return float(self.critic_w @ state)

    def update(self, transitions, actor_lr=1e-2, critic_lr=1e-2, clip_eps=0.2):
        if not transitions:
            raise ValueError("MAPPO update requires transitions")
        actor_grad = np.zeros_like(self.actor_w)
        critic_grad = np.zeros_like(self.critic_w)
        policy_objective = 0.0
        value_loss = 0.0

        for t in transitions:
            obs = np.asarray(t.observation, dtype=float)
            state = np.asarray(t.global_state, dtype=float)
            probs = self.action_probs(obs)
            new_log_prob = float(np.log(max(probs[t.action], 1e-15)))
            ratio = float(np.exp(new_log_prob - t.old_log_prob))
            clipped = float(np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps))
            unclipped_obj = ratio * float(t.advantage)
            clipped_obj = clipped * float(t.advantage)
            use_unclipped = unclipped_obj <= clipped_obj
            objective = min(unclipped_obj, clipped_obj)
            policy_objective += objective

            # Gradient is zero on the flat clipped branch.
            if use_unclipped:
                grad_logp = -probs[:, None] * obs[None, :]
                grad_logp[t.action] += obs
                actor_grad += float(t.advantage) * ratio * grad_logp

            prediction = self.value(state)
            error = prediction - float(t.return_)
            value_loss += 0.5 * error * error
            critic_grad += error * state

        n = float(len(transitions))
        self.actor_w += actor_lr * actor_grad / n
        self.critic_w -= critic_lr * critic_grad / n
        return {
            "policy_objective": policy_objective / n,
            "value_loss": value_loss / n,
        }
