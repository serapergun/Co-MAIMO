from src.coalition.game import allocation_qc, shapley_value
from src.coalition.operational_valuation import (
    OperationalEvidence,
    ValuationWeights,
    operational_gain,
)


def test_operational_gain_reference_and_improvement():
    ref = OperationalEvidence(12, 12, 100.0, 20.0, 10.0)
    better = OperationalEvidence(12, 12, 50.0, 10.0, 5.0)
    assert abs(operational_gain(ref, ref)) < 1e-12
    assert abs(operational_gain(ref, better) - 0.5) < 1e-12


def test_operational_gain_rejects_incomplete_service():
    ref = OperationalEvidence(12, 12, 100.0, 20.0, 10.0)
    bad = OperationalEvidence(11, 12, 50.0, 10.0, 5.0)
    try:
        operational_gain(ref, bad)
        assert False, "incomplete service must be rejected"
    except ValueError:
        pass


def test_shapley_additive_game_and_qc():
    players = ("A", "B", "C")
    worth = {"A": 1.0, "B": 2.0, "C": 4.0}
    values = {}
    for mask in range(1 << len(players)):
        coalition = frozenset(players[i] for i in range(len(players)) if mask & (1 << i))
        values[coalition] = sum(worth[p] for p in coalition)
    phi = shapley_value(players, values)
    assert phi == worth
    qc = allocation_qc(players, values, phi)
    assert qc["efficient"]
    assert qc["individually_rational"]
