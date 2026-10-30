"""Dynamic coalition formation primitives for the B5 benchmark.

Coalitions are accepted only when cooperation is system-beneficial and no
member is made worse off after coordination overhead. This module contains no
fabricated stakeholder utilities; callers must provide counterfactual costs
from executable evidence.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CoalitionProposal:
    members: tuple[str, ...]
    independent_cost: float
    cooperative_cost: float
    coordination_cost: float
    member_independent_costs: dict[str, float]
    member_cooperative_costs: dict[str, float]
    member_coordination_costs: dict[str, float]


@dataclass(frozen=True)
class CoalitionDecision:
    accepted: bool
    coalition_gain: float
    member_gains: dict[str, float]
    reason: str


def evaluate_merge(proposal, tol=1e-12):
    members = tuple(proposal.members)
    if len(members) < 2 or len(set(members)) != len(members):
        raise ValueError("coalition requires at least two unique members")
    if set(proposal.member_independent_costs) != set(members):
        raise ValueError("independent member costs incomplete")
    if set(proposal.member_cooperative_costs) != set(members):
        raise ValueError("cooperative member costs incomplete")
    if set(proposal.member_coordination_costs) != set(members):
        raise ValueError("coordination member costs incomplete")

    coalition_gain = (
        float(proposal.independent_cost)
        - float(proposal.cooperative_cost)
        - float(proposal.coordination_cost)
    )
    member_gains = {
        m: (
            float(proposal.member_independent_costs[m])
            - float(proposal.member_cooperative_costs[m])
            - float(proposal.member_coordination_costs[m])
        )
        for m in members
    }
    if coalition_gain <= tol:
        return CoalitionDecision(False, coalition_gain, member_gains, "no_positive_coalition_gain")
    if any(g < -tol for g in member_gains.values()):
        return CoalitionDecision(False, coalition_gain, member_gains, "individual_rationality_violation")
    return CoalitionDecision(True, coalition_gain, member_gains, "accepted")


def split_required(proposal, tol=1e-12):
    """A formed coalition should split if its current evidence fails merge QC."""
    return not evaluate_merge(proposal, tol=tol).accepted
