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


def test_mappo_masked_action_never_selects_infeasible_taxi():
    import numpy as np
    from src.marl.mappo import LinearMAPPO
    from src.marl.traci_adapter import masked_action

    policy = LinearMAPPO(obs_dim=3, state_dim=1, action_dim=3, seed=1)
    policy.actor_w[:] = 0.0
    rng = np.random.default_rng(2)
    for _ in range(50):
        action, log_prob = masked_action(
            policy,
            np.array([0.2, 0.4, 0.6]),
            np.array([False, True, False]),
            rng,
        )
        assert action == 1
        assert abs(log_prob) < 1e-12

    action, log_prob = masked_action(
        policy,
        np.array([0.2, 0.4, 0.6]),
        np.array([False, False, False]),
        rng,
    )
    assert action is None
    assert log_prob is None


def test_mappo_training_trajectory_uses_discounted_service_returns():
    import numpy as np
    from src.marl.mappo import LinearMAPPO
    from src.marl.training import build_transitions, discounted_returns, dispatch_reward

    rewards = [
        dispatch_reward(20.0, 0.0),
        dispatch_reward(10.0, 5.0),
    ]
    returns = discounted_returns(rewards, gamma=0.5)
    assert abs(returns[0] - (rewards[0] + 0.5 * rewards[1])) < 1e-12
    assert abs(returns[1] - rewards[1]) < 1e-12

    policy = LinearMAPPO(obs_dim=2, state_dim=3, action_dim=2, seed=3)
    decisions = [
        {"observation": [0.2, 1.0], "action": 0, "log_prob": -0.4},
        {"observation": [1.0, 0.3], "action": 1, "log_prob": -0.5},
    ]
    states = [
        np.array([1.0, 0.5, 0.0]),
        np.array([0.5, 1.0, 0.2]),
    ]
    transitions = build_transitions(decisions, rewards, states, policy, gamma=0.5)
    assert len(transitions) == 2
    assert abs(transitions[0].return_ - returns[0]) < 1e-12
    assert abs(transitions[1].return_ - returns[1]) < 1e-12


def test_b6_shapley_feedback_changes_only_coalition_event_rewards():
    from src.marl.shapley_training import b6_rewards
    decisions=[
        {"reservation_id":"r0"},
        {"reservation_id":"r1"},
        {"reservation_id":"r2"},
    ]
    events=[{
        "accepted":True,
        "members":["taxi0","taxi1"],
        "reservation_ids":["r0","r1"],
        "coalition_gain_s":20.0,
        "member_gains_s":{"taxi0":12.0,"taxi1":8.0},
    }]
    base=[-10.0,-20.0,-30.0]
    shaped=b6_rewards(decisions,base,events,lambda_phi=0.1)
    assert shaped[0] != base[0]
    assert shaped[1] != base[1]
    assert shaped[2] == base[2]


def test_masked_ppo_old_new_logprob_consistency_and_reward_signal():
    import numpy as np
    from src.marl.mappo import LinearMAPPO, Transition
    from src.marl.training import dispatch_reward

    policy=LinearMAPPO(obs_dim=3,state_dim=2,action_dim=3,seed=17)
    obs=np.asarray([0.2,0.7,1.0])
    mask=np.asarray([True,False,True])
    probs=policy.masked_action_probs(obs,mask)
    action=2
    old_log=float(np.log(probs[action]))
    # Unchanged policy must reproduce the exact masked likelihood.
    assert abs(float(np.log(policy.masked_action_probs(obs,mask)[action]))-old_log)<1e-12

    before=policy.actor_w.copy()
    t=Transition(
        observation=obs,
        global_state=np.asarray([0.4,0.6]),
        action=action,
        old_log_prob=old_log,
        advantage=2.0,
        return_=1.0,
        action_mask=mask,
    )
    policy.update([t],actor_lr=0.01,critic_lr=0.0)
    # Masked-out action receives no actor gradient.
    assert np.allclose(policy.actor_w[1],before[1])
    assert not np.allclose(policy.actor_w[0],before[0])
    assert not np.allclose(policy.actor_w[2],before[2])

    assert dispatch_reward(60.0,20.0) == -80.0
    assert dispatch_reward(600.0,200.0) == -800.0
    assert dispatch_reward(600.0,200.0) < dispatch_reward(60.0,20.0)
