"""B6 Shapley-informed feedback for accepted operational coalition events.

This is event-level attribution among the taxis participating in an accepted
DCF coalition. It is deliberately separate from the stakeholder
characteristic-game Shapley/nucleolus analysis.
"""

from src.coalition.game import shapley_value


def pair_swap_shapley(taxi_a, taxi_b, gain_a, gain_b, coalition_gain):
    """Exact two-member Shapley allocation of an observed coalition gain.

    Singleton values are the observed individually realizable gains; the grand
    coalition value is the DCF-observed system gain. No infeasible capability
    subsets are invented.
    """
    players=(str(taxi_a), str(taxi_b))
    values={
        frozenset(): 0.0,
        frozenset({players[0]}): float(gain_a),
        frozenset({players[1]}): float(gain_b),
        frozenset(players): float(coalition_gain),
    }
    phi=shapley_value(players, values)
    if abs(sum(phi.values())-float(coalition_gain)) > 1e-9:
        raise RuntimeError("event Shapley efficiency failure")
    return phi


def normalized_feedback(allocation, coalition_gain):
    """Return dimensionless event shares for reward feedback."""
    gain=float(coalition_gain)
    if gain <= 0:
        raise ValueError("Shapley feedback requires positive accepted coalition gain")
    shares={p: float(v)/gain for p,v in allocation.items()}
    if abs(sum(shares.values())-1.0) > 1e-9:
        raise RuntimeError("normalized Shapley shares do not sum to one")
    return shares
