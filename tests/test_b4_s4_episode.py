import pytest
from types import SimpleNamespace
from src.marl.b4_s4_episode import b4_s4_episode_batch
from src.marl.b4_s4_training import TrainableB4S4Actor

def test_delayed_reward_uses_actual_pickup_not_estimated_route():
    dispatcher=SimpleNamespace(decisions=[{
        "reservation_id":"r1","selected_access_time_s":5,
        "candidate_features":[[.2,0,.5,1,1],[.4,0,.6,1,1]],
        "feasible_mask":[1,1],"action":0,"old_log_prob":-.6,
        "global_state":[.5,.5,.2]}])
    qc={"request_first_seen_s":{"r1":10},
        "request_assigned_s":{"r1":20},
        "request_pickup_s":{"r1":50}}
    batch=b4_s4_episode_batch(dispatcher,qc,TrainableB4S4Actor())
    assert len(batch)==1
    assert batch[0]["return_"]==pytest.approx(-.04)

def test_missing_pickup_is_rejected():
    dispatcher=SimpleNamespace(decisions=[{"reservation_id":"r1"}])
    with pytest.raises(ValueError):
        b4_s4_episode_batch(dispatcher,{"request_first_seen_s":{}},TrainableB4S4Actor())
