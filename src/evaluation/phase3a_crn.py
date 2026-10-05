"""Held-out Common Random Number plan for B4-B6 evaluation.

Training seeds and evaluation seeds are disjoint by construction. The same
held-out demand seeds must be reused across B4, B5 and B6 for paired
comparisons.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SeedPlan:
    training: tuple[int, ...]
    heldout: tuple[int, ...]

    def validate(self):
        if not self.training or not self.heldout:
            raise ValueError("training and held-out seed sets must be non-empty")
        if len(set(self.training)) != len(self.training):
            raise ValueError("duplicate training seed")
        if len(set(self.heldout)) != len(self.heldout):
            raise ValueError("duplicate held-out seed")
        overlap = set(self.training) & set(self.heldout)
        if overlap:
            raise ValueError(f"seed leakage between training and held-out sets: {sorted(overlap)}")
        return self


DEFAULT_PHASE3A_SEEDS = SeedPlan(
    training=(4201, 4202, 4203, 4204, 4205, 4206, 4207, 4208),
    heldout=(9101, 9102, 9103, 9104, 9105),
).validate()


def paired_evaluation_rows(methods=("B4", "B5", "B6"), plan=DEFAULT_PHASE3A_SEEDS):
    plan.validate()
    return [
        {"method": method, "demand_seed": seed, "split": "heldout"}
        for seed in plan.heldout
        for method in methods
    ]
