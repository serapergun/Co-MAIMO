"""Numerically stabilized shared-score PPO: reward scaling and clipped gradients.

Training-only changes. Heldout policy remains frozen. The access-time scorer
is permutation equivariant and uses the existing feasibility mask.
"""
import numpy as np
from src.marl.shared_score_mappo import SharedScoreMAPPO
from src.marl.episode import episode_training_batch

REWARD_SCALE_S=1000.0

def train_stabilized_episode(dispatcher,request_qc_path,policy,epochs=4,
                             actor_lr=0.5,critic_lr=0.01,clip_eps=0.2):
    if epochs<1:raise ValueError("epochs must be positive")
    transitions=episode_training_batch(dispatcher,request_qc_path,policy)
    if not transitions:raise ValueError("no transitions")
    # Transform the reward-derived discounted return and baseline into units
    # of 1000 seconds. Critic parameters are trained in these units.
    # episode_training_batch computed advantages using the scaled critic,
    # therefore recompute advantage explicitly rather than scaling the old one.
    for t in transitions:
        t.return_=float(t.return_)/REWARD_SCALE_S
        t.advantage=float(t.return_-policy.value(t.global_state))
    initial_loss=float(np.mean([(t.return_-policy.value(t.global_state))**2 for t in transitions]))
    metrics=None
    for _ in range(epochs):
        metrics=policy.update(transitions,actor_lr=actor_lr,critic_lr=critic_lr,clip_eps=clip_eps)
    return {**metrics,"transition_count":len(transitions),
            "ppo_epochs":epochs,"return_scale_seconds":REWARD_SCALE_S,
            "initial_mean_squared_value_error":initial_loss,
            "actor_coefficient":float(policy.actor_w[0])}
