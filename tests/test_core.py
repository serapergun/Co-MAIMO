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


def test_linear_mappo_action_and_update():
    import numpy as np
    from src.marl.mappo import LinearMAPPO, Transition

    policy = LinearMAPPO(obs_dim=3, state_dim=4, action_dim=2, seed=7)
    probs = policy.action_probs(np.array([1.0, 0.5, -0.25]))
    assert probs.shape == (2,)
    assert abs(float(probs.sum()) - 1.0) < 1e-12
    assert np.all(probs > 0)

    rng = np.random.default_rng(11)
    action, old_log_prob = policy.act(np.array([1.0, 0.5, -0.25]), rng)
    before_actor = policy.actor_w.copy()
    before_critic = policy.critic_w.copy()
    metrics = policy.update([
        Transition(
            observation=np.array([1.0, 0.5, -0.25]),
            global_state=np.array([1.0, 0.0, 0.5, -0.5]),
            action=action,
            old_log_prob=old_log_prob,
            advantage=1.0,
            return_=2.0,
        )
    ])
    assert metrics["value_loss"] >= 0.0
    assert not np.allclose(policy.actor_w, before_actor)
    assert not np.allclose(policy.critic_w, before_critic)
