"""Episode-level MAPPO training orchestration.

This module joins dispatch decisions with completed SUMO lifecycle telemetry
after an episode finishes. It never fabricates pickup outcomes at decision
time.
"""

import json
from pathlib import Path
import numpy as np

from .training import build_transitions, dispatch_reward


def episode_training_batch(dispatcher, request_qc_path, policy, gamma=0.99):
    qc = json.loads(Path(request_qc_path).read_text(encoding="utf-8"))
    first_seen = qc.get("request_first_seen_s", {})
    assigned = qc.get("request_assigned_s", {})
    pickup = qc.get("request_pickup_s", {})

    rewards = []
    states = []
    decisions = []
    for decision in dispatcher.decisions:
        rid = str(decision["reservation_id"])
        if rid not in first_seen or rid not in assigned or rid not in pickup:
            raise ValueError(f"incomplete lifecycle telemetry for reservation {rid}")
        assignment_delay = float(assigned[rid]) - float(first_seen[rid])
        access_time = float(pickup[rid]) - float(assigned[rid])
        if assignment_delay < 0 or access_time < 0:
            raise ValueError(f"invalid lifecycle timestamp ordering for reservation {rid}")

        rewards.append(dispatch_reward(access_time, assignment_delay))
        # Centralized critic state: normalized system-level service context.
        # These quantities are observed from the completed episode, not guessed.
        state = np.asarray([
            assignment_delay / max(1.0, assignment_delay + access_time),
            access_time / max(1.0, assignment_delay + access_time),
            float(len(qc.get("final_active_reservations", []))),
        ], dtype=float)
        if state.shape != (policy.state_dim,):
            raise ValueError(
                f"episode critic state has dimension {len(state)}; "
                f"policy expects {policy.state_dim}"
            )
        states.append(state)
        decisions.append(decision)

    return build_transitions(decisions, rewards, states, policy, gamma=gamma)


def train_completed_episode(dispatcher, request_qc_path, policy, gamma=0.99,
                            actor_lr=1e-2, critic_lr=1e-2, clip_eps=0.2):
    transitions = episode_training_batch(
        dispatcher, request_qc_path, policy, gamma=gamma
    )
    metrics = policy.update(
        transitions,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        clip_eps=clip_eps,
    )
    return {
        **metrics,
        "transition_count": len(transitions),
    }
