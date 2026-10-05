"""Executable benchmark capability registry for Phase3A.

This module separates the documented B1-B6 benchmark labels from what is
actually executable in the repository.  It deliberately refuses to mark a
benchmark executable unless all required implementation components exist.

Capability configurations are nested ablations, not unrestricted cooperative-
game players.  Standard Shapley/nucleolus analysis must therefore use the
separate stakeholder characteristic-game pipeline.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkCapability:
    benchmark: str
    required_components: tuple[str, ...]
    executable: bool
    role: str


BENCHMARKS = {
    "B1": BenchmarkCapability(
        "B1", ("sequential_dispatch",), True,
        "deterministic sequential-dispatch baseline",
    ),
    "B2": BenchmarkCapability(
        "B2", ("static_heterogeneous_allocation",), True,
        "deterministic heterogeneous-allocation baseline",
    ),
    "B3": BenchmarkCapability(
        "B3", ("demand_threshold_heuristic",), True,
        "deterministic demand-threshold baseline",
    ),
    "B4": BenchmarkCapability(
        "B4", ("mappo_policy", "mappo_training", "sumo_policy_adapter"), False,
        "MAPPO benchmark; current barcelona_b4_traci.py is a load fixture only",
    ),
    "B5": BenchmarkCapability(
        "B5", ("mappo_policy", "mappo_training", "sumo_policy_adapter", "dcf", "cce"), False,
        "MAPPO plus dynamic coalition formation and counterfactual coalition evaluation",
    ),
    "B6": BenchmarkCapability(
        "B6", (
            "mappo_policy", "mappo_training", "sumo_policy_adapter",
            "dcf", "cce", "shapley_feedback",
        ), False,
        "B5 plus Shapley-informed reward feedback",
    ),
}


def benchmark_status(name):
    try:
        return BENCHMARKS[str(name)]
    except KeyError as exc:
        raise ValueError(f"Unknown benchmark: {name}") from exc


def require_executable(name):
    spec = benchmark_status(name)
    if not spec.executable:
        missing = ", ".join(spec.required_components)
        raise RuntimeError(
            f"{spec.benchmark} is not yet executable as documented; "
            f"required capability chain: {missing}"
        )
    return spec
