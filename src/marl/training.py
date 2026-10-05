"""Trajectory utilities for training the B4 MAPPO dispatcher.

Training rewards are decision-local and observable at dispatch time. Final
energy claims are intentionally excluded here and must come from held-out SUMO
MMPEVEM outputs.
"""

from dataclasses import dataclass
import numpy as np

from .mappo import Transition


@dataclass(frozen=True)
class RewardConfig:
    access_time_weight: float = 1.0
    queue_delay_weight: float = 1.0

    def validate(self):
        if self.access_time_weight < 0 or self.queue_delay_weight < 0:
            raise ValueError("reward weights must be non-negative")
        if self.access_time_weight + self.queue_delay_weight <= 0:
            raise ValueError("at least one reward weight must be positive")


def dispatch_reward(access_time_s, assignment_delay_s, config=RewardConfig()):
    """Negative normalized service cost for one dispatch decision."""
    config.validate()
    if access_time_s < 0 or assignment_delay_s < 0:
        raise ValueError("dispatch times must be non-negative")
    scale = max(1.0, access_time_s + assignment_delay_s)
    cost = (
        config.access_time_weight * access_time_s
        + config.queue_delay_weight * assignment_delay_s
    ) / scale
    return -float(cost)


def discounted_returns(rewards, gamma=0.99):
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0,1]")
    out = np.zeros(len(rewards), dtype=float)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = float(rewards[i]) + gamma * running
        out[i] = running
    return out


def build_transitions(decisions, rewards, global_states, policy, gamma=0.99):
    """Build PPO transitions with critic-baseline advantages."""
    if not (len(decisions) == len(rewards) == len(global_states)):
        raise ValueError("trajectory lengths differ")
    returns = discounted_returns(rewards, gamma)
    transitions = []
    for decision, reward_return, state in zip(decisions, returns, global_states):
        obs = np.asarray(decision["observation"], dtype=float)
        state = np.asarray(state, dtype=float)
        value = policy.value(state)
        transitions.append(Transition(
            observation=obs,
            global_state=state,
            action=int(decision["action"]),
            old_log_prob=float(decision["log_prob"]),
            advantage=float(reward_return - value),
            return_=float(reward_return),
        ))
    return transitions
