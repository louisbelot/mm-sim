import numpy as np
import pytest

from mmsim import arl
from mmsim import robust as r
from mmsim.ppo import PPO, Config

SMALL = r.Game(T=0.1, Q=5)


class _const:
    def __init__(self, a):
        self.a = a

    def act(self, obs, rng):
        return self.a, 0.0, 0.0


def test_menu_holds_the_equilibrium_corners():
    m = arl.menu(r.Game(), "bAk")
    assert len(m.b) == 12
    assert {(5.0, 105.0, 1.875), (-5.0, 105.0, 1.875), (0.0, 105.0, 1.875)} <= set(
        zip(*m, strict=True)
    )
    assert set(zip(*arl.menu(r.Game(), "b"), strict=True)) == {
        (b, 140.0, 1.5) for b in (-5.0, 0.0, 5.0)
    }


def test_duel_rejects_unknown_side():
    with pytest.raises(ValueError):
        arl.Duel(SMALL, "taker")


def test_duel_sides_are_zero_sum():
    mm, adv = arl.Duel(SMALL, "mm"), arl.Duel(SMALL, "adv")
    total = 0.0
    for seed in range(5):
        mm.reset(seed=seed)
        adv.reset(seed=seed)
        rng = np.random.default_rng(seed)
        for _ in range(SMALL.N):
            a_mm, a_adv = int(rng.integers(81)), int(rng.integers(12))
            mm.other = _const(a_adv)
            adv.other = _const(a_mm)
            _, r1, d1, _, _ = mm.step(a_mm)
            _, r2, d2, _, _ = adv.step(a_adv)
            assert r1 == -r2 and d1 == d2
            total += abs(r1)
    assert total > 0


def test_duel_step_matches_the_exact_expected_payoff():
    g = r.Game(T=0.01, Q=5)
    env = arl.Duel(g, "mm", _const(0), "bAk")
    b, A, k = (float(x[0]) for x in env.opts)
    a = 2 * 9 + 5
    rewards = []
    for seed in range(40_000):
        env.reset(seed=seed)
        env.q = 2
        rewards.append(env.step(a)[1])
    st = r.stage(g, g.terminal()).at(np.array([g.Q + 2]))
    want = r.payoff(st, arl.DELTAS[2], arl.DELTAS[5], b, A, k)[0]
    se = np.std(rewards) / np.sqrt(len(rewards))
    assert abs(np.mean(rewards) - want) < 4 * se


def test_duel_keeps_inventory_inside_limits():
    env = arl.Duel(SMALL, "mm", start=SMALL.Q)
    for seed in range(20):
        env.reset(seed=seed)
        done = False
        while not done:
            _, _, done, _, info = env.step(0)
            assert abs(info["q"]) <= SMALL.Q


def test_tables_have_game_shape_and_grid_values():
    pol = PPO(2, 81, seed=0).policy
    bid, ask = arl.quotes(pol, SMALL)
    assert bid.shape == (SMALL.N, len(SMALL.q)) == ask.shape
    assert set(np.unique(bid)) <= set(arl.DELTAS)
    mkt = arl.market(PPO(2, 3, seed=1).policy, SMALL, "b")
    assert set(np.unique(mkt.b)) <= {-5.0, 0.0, 5.0} and np.all(mkt.A == 140)


def test_rarl_runs_both_sides():
    seen = []
    mm, adv = arl.rarl(SMALL, 2, "b", 1, 1, Config(episodes=2), log=seen.append)
    assert [x.side for x in seen] == ["mm", "adv", "mm", "adv"]
    assert adv is not None and mm.policy.pi.W[-1].shape[1] == 81
    mm, adv = arl.rarl(SMALL, 1, "", 2, 2, Config(episodes=2))
    assert adv is None
