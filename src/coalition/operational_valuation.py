"""Operational evidence adapter for cooperative-game valuation.

This module keeps the SUMO measurement layer separate from the normative
choice of coalition weights.  Metrics are converted to dimensionless costs
relative to an explicit reference scenario; no arbitrary monetary values are
introduced here.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class OperationalEvidence:
    completed: int
    requests: int
    mean_pickup_wait_s: float
    empty_km: float
    net_energy_kWh: float

    @property
    def service_feasible(self):
        return self.completed == self.requests


@dataclass(frozen=True)
class ValuationWeights:
    pickup_wait: float = 1.0
    empty_distance: float = 1.0
    energy: float = 1.0

    def normalized(self):
        total = self.pickup_wait + self.empty_distance + self.energy
        if total <= 0:
            raise ValueError("At least one valuation weight must be positive")
        return (
            self.pickup_wait / total,
            self.empty_distance / total,
            self.energy / total,
        )


def relative_operational_cost(evidence, reference, weights=ValuationWeights()):
    """Dimensionless cost relative to a fixed operational reference.

    Infeasible service is rejected rather than hidden by a finite penalty.
    Each positive reference metric defines one normalized cost component.
    """
    if not evidence.service_feasible:
        raise ValueError("Operational evidence is infeasible: incomplete service")
    if not reference.service_feasible:
        raise ValueError("Reference evidence must be service-feasible")

    denominators = (
        reference.mean_pickup_wait_s,
        reference.empty_km,
        reference.net_energy_kWh,
    )
    if any(x <= 0 for x in denominators):
        raise ValueError("Reference metrics must be strictly positive")

    w_wait, w_empty, w_energy = weights.normalized()
    return (
        w_wait * evidence.mean_pickup_wait_s / reference.mean_pickup_wait_s
        + w_empty * evidence.empty_km / reference.empty_km
        + w_energy * evidence.net_energy_kWh / reference.net_energy_kWh
    )


def operational_gain(reference, cooperative, weights=ValuationWeights()):
    """Positive value means the cooperative case improves on the reference."""
    return relative_operational_cost(reference, reference, weights) - relative_operational_cost(
        cooperative, reference, weights
    )
