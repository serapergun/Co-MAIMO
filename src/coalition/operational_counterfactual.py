"""Operational counterfactual cost contract used by B5 DCF/CCE dispatch.

Costs are derived from executable route evidence only. The cooperative option
may share pickup responsibility between two idle taxis, but no coalition value
is accepted without explicit independent and cooperative route costs.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PairCounterfactual:
    taxi_a: str
    taxi_b: str
    independent_a_s: float
    independent_b_s: float
    cooperative_a_s: float
    cooperative_b_s: float
    coordination_a_s: float = 0.0
    coordination_b_s: float = 0.0

    def validate(self):
        values = (
            self.independent_a_s, self.independent_b_s,
            self.cooperative_a_s, self.cooperative_b_s,
            self.coordination_a_s, self.coordination_b_s,
        )
        if any(float(x) < 0 for x in values):
            raise ValueError("counterfactual route costs must be non-negative")
        if self.taxi_a == self.taxi_b:
            raise ValueError("pair counterfactual requires distinct taxis")
        return self

    @property
    def independent_cost(self):
        return float(self.independent_a_s + self.independent_b_s)

    @property
    def cooperative_cost(self):
        return float(self.cooperative_a_s + self.cooperative_b_s)

    @property
    def coordination_cost(self):
        return float(self.coordination_a_s + self.coordination_b_s)
