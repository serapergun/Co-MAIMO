"""Causal B6 reward shaping from prior accepted DCF events."""

from src.marl.reward import cooperative_reward
from src.marl.shapley_credit import ShapleyCreditMemory
from src.marl.shapley_training import event_shares_by_taxi


def causal_b6_rewards(decisions, base_rewards, coalition_events,
                      lambda_phi=0.1, decay=0.9):
    if len(decisions) != len(base_rewards):
        raise ValueError("decision/reward length mismatch")
    memory=ShapleyCreditMemory(decay=decay)
    events=sorted(
        [e for e in coalition_events if e.get("accepted")],
        key=lambda e: float(e["time"]),
    )
    event_index=0
    shaped=[]
    audit=[]

    for decision,base in zip(decisions,base_rewards):
        now=float(decision["time"])
        # Strict inequality prevents a post-policy event generated at this same
        # dispatch instant from shaping the decision that caused it.
        while event_index < len(events) and float(events[event_index]["time"]) < now:
            event=events[event_index]
            memory.update_from_event(event,event_shares_by_taxi(event))
            event_index+=1

        taxi=str(decision["vehicle_id"])
        credit=memory.credit_for(taxi)
        has_history=taxi in memory.credits
        shaped_reward=cooperative_reward(
            float(base),
            coalition_event=has_history,
            normalized_shapley=credit,
            lambda_phi=float(lambda_phi),
        )
        shaped.append(shaped_reward)
        audit.append({
            "time":now,
            "reservation_id":str(decision["reservation_id"]),
            "vehicle_id":taxi,
            "prior_shapley_credit":credit,
            "has_prior_credit":has_history,
            "base_reward":float(base),
            "shaped_reward":float(shaped_reward),
        })
    return shaped,audit
