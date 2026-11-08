import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from mmsim.accounting import exposure
from mmsim.book import Side
from mmsim.env import DEPTHS, Learned, MarketMakingEnv, sky
from mmsim.informed import Fundamental, InformedTrader
from mmsim.sim import Simulator


def play(env, actions, seed):
    obs, _ = env.reset(seed=seed)
    out = [obs]
    rewards = []
    for a in actions:
        obs, r, done, _, info = env.step(a)
        out.append(obs)
        rewards.append(r)
    return np.array(out), np.array(rewards), info, done


def test_passes_the_gymnasium_checker():
    check_env(MarketMakingEnv(minutes=5.0), skip_render_check=True)


def test_rewards_add_up_to_the_penalized_pnl():
    env = MarketMakingEnv(minutes=40.0, phi=0.03, terminal=0.4)
    actions = np.random.default_rng(0).choice([0, 1, 4, 5], 40)
    _, rewards, info, done = play(env, actions, 7)
    acct = env.ctx.account
    start = env.sim.t - 40.0
    want = info["pnl"] - 0.03 * exposure(acct.fills, start, env.sim.t) - 0.4 * acct.position**2
    assert done
    assert rewards.sum() == pytest.approx(want)
    assert info["fills"] > 5


def test_agent_replays_the_env_episode():
    env = MarketMakingEnv(minutes=30.0, Q=3)
    actions = list(np.random.default_rng(1).integers(0, 16, 30))
    obs, _, _, _ = play(env, actions, 11)

    seen = []
    script = iter(actions)

    def decide(o):
        seen.append(o)
        return next(script, 0)

    bot = Learned(decide, 3, 30.0, 1.0)
    sim = Simulator(sky(), [bot], seed=11)
    sim.seed(10_000, 10_001)
    sim.warm_up(30.0)
    sim.run(30.0)
    assert sim.contexts[0].account.fills == env.ctx.account.fills
    assert np.allclose(np.array(seen[:30]), obs[:30])


def test_one_sided_actions():
    k = len(DEPTHS)
    sell_only = DEPTHS.index(None) * k + 0
    env = MarketMakingEnv(minutes=30.0, Q=50)
    _, _, info, _ = play(env, [sell_only] * 30, 3)
    fills = env.ctx.account.fills
    assert fills and all(f.side == Side.SELL for f in fills)
    _, _, info, _ = play(env, [k * k - 1] * 30, 3)
    assert info["fills"] == 0


def test_position_limit_holds():
    env = MarketMakingEnv(minutes=60.0, Q=2)
    buy_only = 0 * len(DEPTHS) + DEPTHS.index(None)
    _, _, info, _ = play(env, [buy_only] * 60, 5)
    assert info["position"] == 2
    q = np.cumsum([f.qty if f.side == Side.BUY else -f.qty for f in env.ctx.account.fills])
    assert q.max() == 2


def test_observations_stay_in_the_space():
    env = MarketMakingEnv(minutes=60.0)
    actions = np.random.default_rng(2).integers(0, 16, 60)
    obs, _, _, _ = play(env, actions, 9)
    assert all(env.observation_space.contains(o) for o in obs)
    assert np.allclose(obs[:, 1], np.linspace(1, 0, 61))


def test_quotes_sit_at_the_chosen_depths():
    env = MarketMakingEnv(minutes=10.0)
    env.reset(seed=4)
    a = DEPTHS.index(2) * len(DEPTHS) + DEPTHS.index(1)
    env.step(a)
    ctx = env.ctx
    env.puppet.requote(ctx)
    live = {o.side: o.price for o in ctx.orders.values()}
    assert live == {Side.BUY: ctx.bid - 2, Side.SELL: ctx.ask + 1}


def test_imbalance_ignores_own_orders():
    env = MarketMakingEnv(minutes=10.0)
    env.reset(seed=4)
    env.step(0)
    ctx, bot = env.ctx, env.puppet
    bot.requote(ctx)
    b = ctx.book.level(Side.BUY, ctx.bid).qty - 1
    a = ctx.book.level(Side.SELL, ctx.ask).qty - 1
    assert bot.observe(ctx)[3] == pytest.approx((b - a) / (b + a))


def test_crowd_joins_the_market():
    def crowd(seed):
        return [InformedTrader(Fundamental(1.0, seed), 2.0, seed=seed)]

    env = MarketMakingEnv(minutes=20.0, crowd=crowd)
    play(env, [0] * 20, 2)
    assert len(env.sim.contexts) == 2
    assert env.sim.contexts[1].account.fills
    assert env.ctx.agent is env.puppet
