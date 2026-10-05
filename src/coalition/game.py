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


def _proper_nonempty_coalitions(players):
    grand = frozenset(players)
    return [s for s in all_coalitions(players) if s and s != grand]


def least_core(players, values, tol=1e-9):
    """Exact least-core solution for small games by active-set enumeration.

    Solves min epsilon subject to efficiency and
    v(S)-x(S) <= epsilon for every proper non-empty coalition S.
    At an LP vertex, efficiency plus n active coalition constraints determine
    (x_1,...,x_n,epsilon). All candidate vertices are checked against every
    coalition inequality; the feasible candidate with minimum epsilon is
    returned. Intended for the 3-4 stakeholder games used in this project.
    """
    import numpy as np

    players = validate_game(players, values)
    n = len(players)
    if n == 0:
        return {}, 0.0
    coalitions = _proper_nonempty_coalitions(players)
    if len(coalitions) < n:
        raise ValueError("Insufficient coalition constraints for least-core enumeration")

    grand = frozenset(players)
    best = None
    for active in combinations(coalitions, n):
        a = np.zeros((n + 1, n + 1), dtype=float)
        b = np.zeros(n + 1, dtype=float)
        a[0, :n] = 1.0
        b[0] = float(values[grand])
        for row, s in enumerate(active, start=1):
            for col, p in enumerate(players):
                if p in s:
                    a[row, col] = 1.0
            a[row, n] = 1.0
            b[row] = float(values[s])
        try:
            z = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            continue
        allocation = {p: float(z[i]) for i, p in enumerate(players)}
        epsilon = float(z[n])
        if all(excess(s, allocation, values) <= epsilon + tol for s in coalitions):
            if best is None or epsilon < best[1] - tol:
                best = (allocation, epsilon)
    if best is None:
        raise RuntimeError("No feasible least-core vertex found")
    return best


def nucleolus(players, values, tol=1e-9):
    """Return the nucleolus when the least-core minimizer is unique.

    For the project's small games this provides an exact, dependency-light
    path. If the first least-core stage has multiple optimal allocations,
    lexicographic refinement is required and the function fails explicitly
    rather than returning a heuristic allocation.
    """
    import numpy as np

    players = validate_game(players, values)
    allocation, epsilon = least_core(players, values, tol=tol)
    n = len(players)
    coalitions = _proper_nonempty_coalitions(players)
    tight = [s for s in coalitions if abs(excess(s, allocation, values) - epsilon) <= tol]

    # Uniqueness: efficiency plus tight excess equalities must determine x.
    rows = [np.ones(n, dtype=float)]
    for s in tight:
        rows.append(np.array([1.0 if p in s else 0.0 for p in players], dtype=float))
    rank = np.linalg.matrix_rank(np.vstack(rows), tol=tol)
    if rank < n:
        raise RuntimeError(
            "Least-core minimizer is not unique; lexicographic nucleolus refinement required"
        )

    qc = allocation_qc(players, values, allocation, tol=tol)
    return allocation, {
        "least_core_epsilon": epsilon,
        "tight_coalitions": tight,
        "rank": int(rank),
        **qc,
    }
