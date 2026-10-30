"""Request state machine and conservation accounting."""
VALID = {"WAITING", "ASSIGNED", "PICKED_UP", "SERVED", "UNSERVED"}

class RequestLedger:
    def __init__(self, request_ids):
        ids = list(request_ids)
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate request IDs.")
        self.state = {r: "WAITING" for r in ids}

    def transition(self, request_id, new_state):
        old = self.state[request_id]
        allowed = {
            "WAITING": {"ASSIGNED", "UNSERVED"},
            "ASSIGNED": {"WAITING", "PICKED_UP", "UNSERVED"},
            "PICKED_UP": {"SERVED"},
            "SERVED": set(),
            "UNSERVED": set(),
        }
        if new_state not in VALID or new_state not in allowed[old]:
            raise ValueError(f"Invalid transition {old}->{new_state} for {request_id}")
        self.state[request_id] = new_state

    def counts(self):
        out = {s: 0 for s in VALID}
        for s in self.state.values(): out[s] += 1
        out["generated"] = len(self.state)
        out["remaining"] = out["WAITING"] + out["ASSIGNED"] + out["PICKED_UP"]
        out["served"] = out["SERVED"]
        out["unserved"] = out["UNSERVED"]
        return out
