"""Phase3A coalition capability contract.

Players are algorithmic capabilities, not fleet-size configurations.  A
coalition scenario is valid only when its enabled capabilities respect their
dependencies. This keeps counterfactual attribution tied to executable model
components rather than stakeholder labels that are not yet implemented.
"""

PLAYERS = ("marl", "dcf", "shapley_feedback")

DEPENDENCIES = {
    "marl": frozenset(),
    "dcf": frozenset({"marl"}),
    "shapley_feedback": frozenset({"marl", "dcf"}),
}


def validate_capability_coalition(coalition):
    s = frozenset(str(x) for x in coalition)
    unknown = s - frozenset(PLAYERS)
    if unknown:
        raise ValueError(f"Unknown Phase3A capability: {sorted(unknown)}")
    for capability in s:
        missing = DEPENDENCIES[capability] - s
        if missing:
            raise ValueError(
                f"{capability} requires capabilities {sorted(missing)}"
            )
    return s


def benchmark_mapping(coalition):
    """Map implemented capability sets to the documented benchmark chain."""
    s = validate_capability_coalition(coalition)
    mapping = {
        frozenset(): "B1-B3 classical baseline family",
        frozenset({"marl"}): "B4",
        frozenset({"marl", "dcf"}): "B5",
        frozenset({"marl", "dcf", "shapley_feedback"}): "B6",
    }
    return mapping[s]
