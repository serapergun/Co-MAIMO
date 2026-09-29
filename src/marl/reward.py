def cooperative_reward(system_reward, coordination_cost=0.0,
                       coalition_event=False, normalized_shapley=0.0,
                       lambda_phi=0.0, lambda_coord=0.0):
    """Event-only Shapley feedback.

    Outside coalition events the agent receives the full system reward.
    """
    reward = system_reward - lambda_coord * coordination_cost
    if coalition_event:
        reward += lambda_phi * (normalized_shapley - system_reward)
    return reward
