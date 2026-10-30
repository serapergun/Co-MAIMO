"""Build a characteristic game from Phase3A coalition counterfactual records."""

from .operational_valuation import OperationalEvidence, ValuationWeights, operational_gain


REQUIRED_METRICS = (
    "completed",
    "requests",
    "mean_pickup_wait_s",
    "empty_km",
    "net_energy_kWh",
)


def _coalition_key(players):
    return frozenset(str(p) for p in players)


def evidence_from_record(record):
    missing = [k for k in REQUIRED_METRICS if k not in record]
    if missing:
        raise ValueError(f"Coalition record missing operational metrics: {missing}")
    return OperationalEvidence(
        completed=int(record["completed"]),
        requests=int(record["requests"]),
        mean_pickup_wait_s=float(record["mean_pickup_wait_s"]),
        empty_km=float(record["empty_km"]),
        net_energy_kWh=float(record["net_energy_kWh"]),
    )


def characteristic_values_from_counterfactuals(players, records, reference,
                                               weights=ValuationWeights()):
    """Convert complete Phase3A counterfactual records into v(S).

    The empty coalition is fixed to zero. Every non-empty coalition must have
    exactly one counterfactual record. Operationally infeasible coalitions are
    rejected: their worth must be handled by an explicit modelling policy,
    never by an implicit fabricated penalty.
    """
    players = tuple(str(p) for p in players)
    allowed = frozenset(players)
    by_coalition = {}
    for record in records:
        if "coalition" not in record:
            raise ValueError("Coalition record missing 'coalition'")
        s = _coalition_key(record["coalition"])
        if not s or not s <= allowed:
            raise ValueError(f"Invalid coalition: {sorted(s)}")
        if s in by_coalition:
            raise ValueError(f"Duplicate coalition record: {sorted(s)}")
        by_coalition[s] = record

    expected = {
        frozenset(players[i] for i in range(len(players)) if mask & (1 << i))
        for mask in range(1, 1 << len(players))
    }
    missing = expected - set(by_coalition)
    extra = set(by_coalition) - expected
    if missing or extra:
        raise ValueError(f"Incomplete counterfactual set: missing={missing}, extra={extra}")

    values = {frozenset(): 0.0}
    for s in expected:
        evidence = evidence_from_record(by_coalition[s])
        values[s] = operational_gain(reference, evidence, weights)
    return values
