"""Causal Shapley credit memory for B6 on-policy reward shaping.

Credits are created only after an accepted DCF event and may affect only later
MAPPO decisions by the same taxi. No future coalition information is propagated
backward into earlier PPO transitions.
"""

from dataclasses import dataclass, field


@dataclass
class ShapleyCreditMemory:
    decay: float = 0.9
    credits: dict = field(default_factory=dict)

    def __post_init__(self):
        if not 0.0 <= self.decay <= 1.0:
            raise ValueError("decay must be in [0,1]")

    def update_from_event(self, event, shares):
        if not event.get("accepted"):
            return
        for taxi in event["members"]:
            share=float(shares[str(taxi)])
            if not 0.0 <= share <= 1.0:
                raise ValueError("normalized Shapley share must be in [0,1]")
            previous=float(self.credits.get(str(taxi),0.5))
            self.credits[str(taxi)]=self.decay*previous+(1.0-self.decay)*share

    def credit_for(self, taxi):
        return float(self.credits.get(str(taxi),0.5))

    def snapshot(self):
        return {str(k):float(v) for k,v in self.credits.items()}
