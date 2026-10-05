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


def test_nucleolus_symmetric_three_player_game():
    from src.coalition.game import nucleolus

    players = ("A", "B", "C")
    values = {
        frozenset(): 0.0,
        frozenset({"A"}): 0.0,
        frozenset({"B"}): 0.0,
        frozenset({"C"}): 0.0,
        frozenset({"A", "B"}): 0.0,
        frozenset({"A", "C"}): 0.0,
        frozenset({"B", "C"}): 0.0,
        frozenset({"A", "B", "C"}): 1.0,
    }
    x, qc = nucleolus(players, values)
    assert all(abs(x[p] - 1.0 / 3.0) < 1e-9 for p in players)
    assert qc["efficient"]
    assert qc["individually_rational"]
    assert qc["rank"] == 3


def test_counterfactual_adapter_requires_complete_game():
    from src.coalition.counterfactual_adapter import characteristic_values_from_counterfactuals
    from src.coalition.operational_valuation import OperationalEvidence

    players = ("A", "B")
    ref = OperationalEvidence(10, 10, 100.0, 20.0, 10.0)
    def rec(coalition, wait, empty, energy):
        return {
            "coalition": coalition,
            "completed": 10,
            "requests": 10,
            "mean_pickup_wait_s": wait,
            "empty_km": empty,
            "net_energy_kWh": energy,
        }
    records = [
        rec(["A"], 90, 18, 9),
        rec(["B"], 80, 16, 8),
        rec(["A", "B"], 50, 10, 5),
    ]
    values = characteristic_values_from_counterfactuals(players, records, ref)
    assert set(values) == {
        frozenset(), frozenset({"A"}), frozenset({"B"}), frozenset({"A", "B"})
    }
    assert abs(values[frozenset({"A", "B"})] - 0.5) < 1e-12

    try:
        characteristic_values_from_counterfactuals(players, records[:-1], ref)
        assert False, "missing coalition must be rejected"
    except ValueError:
        pass


def test_phase3a_exporter_rejects_contaminated_run():
    from scripts.export_phase3a_counterfactual import record_from_summary

    clean = {
        "completed": 12,
        "requests": 12,
        "unresolved": 0,
        "teleports": 0,
        "horizon_exceeded": False,
        "mean_pickup_wait_s": 100.0,
        "empty_km": 20.0,
        "net_energy_kWh": 10.0,
    }
    record = record_from_summary(("operator", "infrastructure"), clean)
    assert record["coalition"] == ["operator", "infrastructure"]
    assert record["completed"] == 12

    contaminated = dict(clean, teleports=1)
    try:
        record_from_summary(("operator",), contaminated)
        assert False, "teleport-contaminated coalition run must be rejected"
    except ValueError:
        pass


def test_phase3a_aggregator_requires_all_coalitions():
    from scripts.aggregate_phase3a_counterfactuals import aggregate_records

    players = ("A", "B", "C")
    coalitions = [
        ["A"], ["B"], ["C"],
        ["A", "B"], ["A", "C"], ["B", "C"],
        ["A", "B", "C"],
    ]
    records = [{"coalition": s, "marker": i} for i, s in enumerate(coalitions)]
    dataset = aggregate_records(players, records)
    assert dataset["complete"] is True
    assert dataset["coalition_count"] == 7
    assert dataset["expected_coalition_count"] == 7

    try:
        aggregate_records(players, records[:-1])
        assert False, "incomplete coalition set must be rejected"
    except ValueError:
        pass


def test_phase3a_analysis_runner_end_to_end():
    from scripts.analyze_phase3a_game import analyze_dataset
    from src.coalition.operational_valuation import OperationalEvidence

    players = ["A", "B"]
    def rec(s, scale):
        return {
            "coalition": s,
            "completed": 10,
            "requests": 10,
            "mean_pickup_wait_s": 100.0 * scale,
            "empty_km": 20.0 * scale,
            "net_energy_kWh": 10.0 * scale,
        }
    dataset = {
        "players": players,
        "coalition_count": 3,
        "expected_coalition_count": 3,
        "complete": True,
        "records": [
            rec(["A"], 0.8),
            rec(["B"], 0.8),
            rec(["A", "B"], 0.5),
        ],
    }
    ref = OperationalEvidence(10, 10, 100.0, 20.0, 10.0)
    result = analyze_dataset(dataset, ref)
    assert abs(result["characteristic_values"]["{A}"] - 0.2) < 1e-12
    assert abs(result["characteristic_values"]["{B}"] - 0.2) < 1e-12
    assert abs(result["characteristic_values"]["{A,B}"] - 0.5) < 1e-12
    assert abs(result["shapley"]["A"] - 0.25) < 1e-12
    assert abs(result["shapley"]["B"] - 0.25) < 1e-12
    assert result["shapley_qc"]["efficient"]
    assert result["nucleolus"] is not None
    assert abs(result["nucleolus"]["A"] - 0.25) < 1e-9
    assert abs(result["nucleolus"]["B"] - 0.25) < 1e-9


def test_phase3a_capability_dependencies_and_benchmarks():
    from src.coalition.capabilities import benchmark_mapping, validate_capability_coalition

    assert benchmark_mapping(()) == "B1-B3 classical baseline family"
    assert benchmark_mapping(("marl",)) == "B4"
    assert benchmark_mapping(("marl", "dcf")) == "B5"
    assert benchmark_mapping(("marl", "dcf", "shapley_feedback")) == "B6"

    for invalid in [("dcf",), ("shapley_feedback",), ("marl", "shapley_feedback")]:
        import pytest
        with pytest.raises(ValueError):
            validate_capability_coalition(invalid)
