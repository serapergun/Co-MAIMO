def cooperative_reward(system_reward, coordination_cost=0.0,
                       coalition_event=False, normalized_shapley=0.0,
                       lambda_phi=0.0, lambda_coord=0.0):
    # Shapley is dimensionless, so it modulates reward magnitude.
    if lambda_phi < 0 or lambda_coord < 0:
        raise ValueError('reward coefficients must be non-negative')
    if coalition_event and not 0.0 <= normalized_shapley <= 1.0:
        raise ValueError('normalized Shapley share must be in [0,1]')
    reward = float(system_reward) - float(lambda_coord) * float(coordination_cost)
    if coalition_event:
        reward += (float(lambda_phi) * abs(float(system_reward))
                   * (float(normalized_shapley) - 0.5))
    return reward
