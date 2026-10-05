"""Exact cooperative-game utilities for small stakeholder games.

Coalitions are represented as frozensets of player labels.  The characteristic
function must include the empty coalition with value zero and every subset of
the declared player set.
"""

from itertools import combinations
from math import factorial


def all_coalitions(players):
    players = tuple(players)
    return [
        frozenset(c)
        for r in range(len(players) + 1)
        for c in combinations(players, r)
    ]


def validate_game(players, values, tol=1e-12):
    players = tuple(players)
    expected = set(all_coalitions(players))
    supplied = set(values)
    missing = expected - supplied
    extra = supplied - expected
    if missing or extra:
        raise ValueError(f"Incomplete characteristic function: missing={missing}, extra={extra}")
    if abs(float(values[frozenset()])) > tol:
        raise ValueError("The empty coalition must have value zero")
    return players


def shapley_value(players, values):
    """Return the exact Shapley allocation for a complete characteristic game."""
    players = validate_game(players, values)
    n = len(players)
    if n == 0:
        return {}
    out = {}
    denom = factorial(n)
    for i in players:
        phi = 0.0
        others = [p for p in players if p != i]
        for r in range(len(others) + 1):
            weight = factorial(r) * factorial(n - r - 1) / denom
            for subset in combinations(others, r):
                s = frozenset(subset)
                phi += weight * (float(values[s | {i}]) - float(values[s]))
        out[i] = phi
    return out


def excess(coalition, allocation, values):
    """Coalition dissatisfaction e(S,x)=v(S)-sum_{i in S} x_i."""
    s = frozenset(coalition)
    return float(values[s]) - sum(float(allocation[i]) for i in s)


def allocation_qc(players, values, allocation, tol=1e-9):
    """Check efficiency and individual rationality for an allocation."""
    players = validate_game(players, values)
    grand = frozenset(players)
    efficient = abs(sum(allocation[p] for p in players) - float(values[grand])) <= tol
    individually_rational = all(
        allocation[p] + tol >= float(values[frozenset({p})]) for p in players
    )
    return {
        "efficient": efficient,
        "individually_rational": individually_rational,
    }
