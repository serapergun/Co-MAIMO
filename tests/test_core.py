from src.environment.bev import BEVState
from src.environment.request_ledger import RequestLedger
from src.coalition.cce import CCEOutcome, coalition_value, individual_gain
from src.marl.reward import cooperative_reward

def test_bev_reserve_guard():
    b = BEVState(60.0, 30.0, 0.10)
    assert b.feasible(20.0)
    assert not b.feasible(25.0)

def test_request_conservation():
    x = RequestLedger(["a","b"])
    x.transition("a","ASSIGNED")
    x.transition("a","PICKED_UP")
    x.transition("a","SERVED")
    x.transition("b","UNSERVED")
    c=x.counts()
    assert c["generated"] == c["served"] + c["unserved"] + c["remaining"]

def test_cce_sign():
    assert coalition_value(CCEOutcome(10), CCEOutcome(7,1)) == 2
    assert individual_gain(5,4,0.5) == 0.5

def test_event_only_shapley_reward():
    assert cooperative_reward(10, coalition_event=False, normalized_shapley=99, lambda_phi=.5) == 10
    assert cooperative_reward(10, coalition_event=True, normalized_shapley=6, lambda_phi=.5) == 8
