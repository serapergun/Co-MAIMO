from dataclasses import dataclass

@dataclass
class CCEOutcome:
    operational_cost: float
    coordination_cost: float = 0.0

def coalition_value(independent, cooperative):
    """Positive iff cooperation reduces cost after coordination overhead."""
    return (
        independent.operational_cost
        - cooperative.operational_cost
        - cooperative.coordination_cost
    )

def individual_gain(individual_independent_cost, individual_coalition_cost,
                    individual_coordination_cost=0.0):
    """CCE-local utility gain used for merge/split individual rationality."""
    return (
        individual_independent_cost
        - individual_coalition_cost
        - individual_coordination_cost
    )
