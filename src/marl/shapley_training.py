"""B6 reward construction from completed lifecycle + accepted DCF events."""

from src.coalition.event_shapley import pair_swap_shapley, normalized_feedback
from src.marl.reward import cooperative_reward


def shapley_feedback_by_reservation(coalition_events):
    feedback={}
    for event in coalition_events:
        if not event.get("accepted"):
            continue
        members=event["members"]
        reservations=event["reservation_ids"]
        gains=event["member_gains_s"]
        phi=pair_swap_shapley(
            members[0], members[1],
            gains[members[0]], gains[members[1]],
            event["coalition_gain_s"],
        )
        shares=normalized_feedback(phi, event["coalition_gain_s"])
        # Accepted DCF swaps rewrite decision reservation_id to the
        # actually served reservation. taxi_a serves reservation_b and vice versa.
        feedback[str(reservations[1])] = shares[members[0]]
        feedback[str(reservations[0])] = shares[members[1]]
    return feedback


def b6_rewards(decisions, base_rewards, coalition_events,
               lambda_phi=0.1, lambda_coord=0.0):
    if len(decisions) != len(base_rewards):
        raise ValueError("decision/reward length mismatch")
    shares=shapley_feedback_by_reservation(coalition_events)
    out=[]
    for decision, base in zip(decisions, base_rewards):
        rid=str(decision["reservation_id"])
        is_event=rid in shares
        out.append(cooperative_reward(
            float(base),
            coordination_cost=0.0,
            coalition_event=is_event,
            normalized_shapley=shares.get(rid, 0.0),
            lambda_phi=float(lambda_phi),
            lambda_coord=float(lambda_coord),
        ))
    return out


def event_shares_by_taxi(event):
    """Return exact normalized taxi shares for one accepted DCF event."""
    if not event.get("accepted"):
        return {}
    members=event["members"]
    gains=event["member_gains_s"]
    phi=pair_swap_shapley(
        members[0],members[1],
        gains[members[0]],gains[members[1]],
        event["coalition_gain_s"],
    )
    shares=normalized_feedback(phi,event["coalition_gain_s"])
    return {str(k):float(v) for k,v in shares.items()}
