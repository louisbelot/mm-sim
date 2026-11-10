import math

import numpy as np
import pytest
from scipy.optimize import minimize

from mmsim.book import Side
from mmsim.bots import GLFT, RegimeLean, SymmetricQuoter
from mmsim.detect import one_sided
from mmsim.execution import (
    AlmgrenChriss,
    Executor,
    ObizhaevaWang,
    Passive,
    Pov,
    fit_response,
    impulse,
    scatter,
    shortfall,
    targets,
    trial,
)
from mmsim.flow import CSTParams, ZeroIntelligence
from mmsim.glft import Model
from mmsim.sim import Agent, Print, Simulator
from mmsim.tournament import session

PAPER = AlmgrenChriss(sigma=0.95, eta=2.5e-6, gamma=2.5e-7, eps=0.0625)
X, T, N = 1e6, 5.0, 5
SKY = CSTParams.sky_perfect()


def flow():
    return ZeroIntelligence(SKY)


def frontier(m, lam):
    tau = T / N
    k = m.kappa(lam, tau)
    eta = m.eta - 0.5 * m.gamma * tau
    sh = np.sinh(k * T)
    E = 0.5 * m.gamma * X**2 + m.eps * X
    E += (
        eta
        * X**2
        * np.tanh(0.5 * k * tau)
        * (tau * np.sinh(2 * k * T) + 2 * T * np.sinh(k * tau))
        / (2 * tau**2 * sh**2)
    )
    V = (
        0.5
        * m.sigma**2
        * X**2
        * (tau * sh * np.cosh(k * (T - tau)) - T * np.sinh(k * tau))
        / (sh**2 * np.sinh(k * tau))
    )
    return E, V


def test_kappa_of_the_paper_example():
    k = PAPER.kappa(1e-6, 1.0)
    assert k == pytest.approx(0.6, abs=0.01)
    assert k * T == pytest.approx(3.0, abs=0.05)


@pytest.mark.parametrize("lam", [1e-8, 1e-6, 2e-6, 1e-4])
def test_cost_matches_the_closed_form_frontier(lam):
    E, V = PAPER.cost(PAPER.holdings(X, T, N, lam), T / N)
    assert (E, V) == pytest.approx(frontier(PAPER, lam), rel=1e-10)


def test_naive_strategies():
    tau = T / N
    E, V = PAPER.cost(PAPER.holdings(X, T, N, 0.0), tau)
    eta = PAPER.eta - 0.5 * PAPER.gamma * tau
    assert E == pytest.approx(0.5 * PAPER.gamma * X**2 + PAPER.eps * X + eta * X**2 / T)
    assert V == pytest.approx(PAPER.sigma**2 * X**2 * T * (1 - 1 / N) * (1 - 1 / (2 * N)) / 3)
    dump = np.array([X, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert PAPER.cost(dump, tau) == pytest.approx((PAPER.eps * X + PAPER.eta * X**2 / tau, 0.0))


@pytest.mark.parametrize("lam", [2e-7, 1e-6, 5e-6])
def test_holdings_minimize_cost_plus_risk(lam):
    m = AlmgrenChriss(PAPER.sigma, PAPER.eta, PAPER.gamma)
    scale = 1e-6

    def u(inner):
        E, V = m.cost(np.concatenate([[X], inner / scale, [0.0]]), T / N)
        return (E + lam * V) * 1e-6

    best = minimize(u, np.linspace(X, 0.0, N + 1)[1:-1] * scale, tol=1e-14)
    assert best.x / scale == pytest.approx(m.holdings(X, T, N, lam)[1:-1], rel=1e-4)


def test_holdings_shape():
    slow = PAPER.holdings(X, T, 50, 1e-12)
    assert slow == pytest.approx(np.linspace(X, 0.0, 51), rel=1e-5, abs=1.0)
    fast = PAPER.holdings(X, T, 50, 1.0)
    assert fast[0] == X and fast[-1] == 0.0
    assert np.all(np.isfinite(fast)) and fast[1] < 1e-3 * X
    mid = PAPER.holdings(X, T, 50, 1e-6)
    assert np.all(np.diff(mid) < 0) and np.all(np.diff(mid, 2) > 0)


@pytest.mark.parametrize(
    "rho, x0",
    [(0.5, 40_000), (1, 33_333), (2, 25_000), (4, 16_667), (5, 14_286), (10, 8_333), (20, 4_545)],
)
def test_ow_table_1(rho, x0):
    x = ObizhaevaWang(1.0, rho).trades(100_000, 1.0, 200_000)
    assert x[0] == x[-1] == pytest.approx(x0, abs=1.0)
    assert x[1:-1].sum() == pytest.approx(100_000 - 2 * x0, abs=2.0)


@pytest.mark.parametrize(
    "rho, row",
    [
        (0.5, (2.82, 5.42, 5.99, 6.06, 6.13)),
        (1, (3.98, 8.16, 9.14, 9.26, 9.39)),
        (2, (4.32, 9.97, 11.51, 11.71, 11.92)),
        (10, (1.13, 4.58, 6.65, 7.01, 7.41)),
        (300, (0.00, 0.02, 0.08, 0.13, 0.33)),
    ],
)
def test_ow_table_2(rho, row):
    q = 5000.0
    for lam, saving in zip(
        (1 / (2 * q), 1 / (10 * q), 1 / (50 * q), 1 / (100 * q), 0.0), row, strict=True
    ):
        m = ObizhaevaWang(1 / q - lam, rho, lam)
        cm, opt = m.steady(100_000, 1.0), m.optimum(100_000, 1.0)
        assert 100 * (cm - opt) / cm == pytest.approx(saving, abs=0.0051)


def test_ow_trades_solve_the_quadratic_program():
    m = ObizhaevaWang(0.3, 1.7, 0.05)
    t = np.linspace(0.0, 2.0, 9)
    K = np.exp(-m.rho * np.abs(np.subtract.outer(t, t)))
    w = np.linalg.solve(K, np.ones(9))
    x = m.trades(50.0, 2.0, 8)
    assert x == pytest.approx(50.0 * w / w.sum(), rel=1e-12)
    rng = np.random.default_rng(0)
    for _ in range(20):
        d = rng.normal(size=9)
        assert m.cost(x + d - d.mean(), t) > m.cost(x, t)


def test_ow_cost_limits():
    m = ObizhaevaWang(0.3, 1.7, 0.05)
    n = 4000
    t = np.linspace(0.0, 2.0, n + 1)
    assert m.cost(m.trades(50.0, 2.0, n), t) == pytest.approx(m.optimum(50.0, 2.0), rel=1e-3)
    even = np.full(n + 1, 50.0 / (n + 1))
    assert m.cost(even, t) == pytest.approx(m.steady(50.0, 2.0), rel=1e-3)
    block = m.cost(np.array([50.0]), np.array([0.0]))
    assert block == pytest.approx(0.5 * (m.lam + m.kappa) * 50.0**2)


def test_ow_cost_by_walking_the_book():
    m = ObizhaevaWang(0.4, 0.9, 0.1)
    x = np.array([3.0, 1.0, 2.5])
    t = np.array([0.0, 0.7, 1.5])
    perm, d, last, total = 0.0, 0.0, 0.0, 0.0
    for xi, ti in zip(x, t, strict=True):
        d *= np.exp(-m.rho * (ti - last))
        total += xi * (perm + d + 0.5 * (m.lam + m.kappa) * xi)
        perm += m.lam * xi
        d += m.kappa * xi
        last = ti
    assert m.cost(x, t) == pytest.approx(total)


def test_fit_response_recovers_the_model():
    truth = ObizhaevaWang(0.8, 0.35, 0.15)
    lags = np.array([0.0, 0.5, 1, 2, 4, 8, 16, 32])
    fit = fit_response(lags, truth.response(lags), np.ones(len(lags)))
    assert (fit.kappa, fit.rho, fit.lam) == pytest.approx((0.8, 0.35, 0.15), rel=1e-5)


def test_targets_round_without_losing_shares():
    done = targets(PAPER.holdings(37, 1.0, 12, 30.0))
    assert done[0] == 0 and done[-1] == 37
    assert np.all(np.diff(done) >= 0)


def run(agent, minutes, seed=0, warm=0.0):
    sim = Simulator(flow(), [agent], seed=seed)
    sim.seed(10_000, 10_001)
    sim.warm_up(warm)
    sim.run(minutes)
    return sim.contexts[0]


def test_block_walks_the_seeded_book():
    bot = Executor(Side.BUY, [0.0], [5])
    ctx = run(bot, 0.0)
    assert bot.arrival == 10_000.5
    assert [(f.price, f.qty) for f in ctx.account.fills] == [(10_001, 1)] * 3 + [(10_002, 1)] * 2
    assert shortfall(ctx.account.fills, bot.arrival) == pytest.approx(3 * 0.5 + 2 * 1.5)


def test_seller_pays_too():
    bot = Executor(Side.SELL, [0.0], [5])
    ctx = run(bot, 0.0)
    assert ctx.position == -5
    assert shortfall(ctx.account.fills, bot.arrival) == pytest.approx(4.5)


def test_executor_follows_the_schedule():
    times = [0.0, 10.0, 20.0, 30.0]
    bot = Executor(Side.BUY, times, [2, 4, 4, 9])
    ctx = run(bot, 40.0, seed=3, warm=30.0)
    assert ctx.position == 9
    got = {}
    for f in ctx.account.fills:
        got[round(f.t - 30.0, 9)] = got.get(round(f.t - 30.0, 9), 0) + f.qty
    assert got == {0.0: 2, 10.0: 2, 30.0: 5}


def test_executor_catches_up_after_a_partial_fill():
    bot = Executor(Side.BUY, [0.0, 5.0], [60, 60])
    ctx = run(bot, 10.0)
    first = sum(f.qty for f in ctx.account.fills if f.t == 0.0)
    assert first < 60
    assert ctx.position == first + sum(f.qty for f in ctx.account.fills if f.t == 5.0)
    assert ctx.position > first


def test_impulse_is_zero_without_a_trade():
    assert not impulse(flow, 0, [0.0, 1.0, 5.0], 1).any()


def test_impulse_moves_the_ask_first():
    d = np.mean([impulse(flow, 3, [0.0, 2.0], s) for s in range(40)], axis=0)
    bid0, ask0 = d[0]
    assert bid0 == 0.0 and ask0 >= 1.0
    assert d[1, 1] < ask0


def test_arrival_is_the_mid_at_the_first_slice():
    late = Executor(Side.BUY, [12.0], [1])
    sim = Simulator(flow(), [late], seed=5)
    sim.seed(10_000, 10_001)
    sim.run(20.0)
    assert late.arrival == sim.tape.at(12.0 - 1e-9)
    assert late.arrival != 10_000.5


def test_arrival_falls_back_to_the_tape_on_a_one_sided_book():
    bot = Pov(Side.BUY, 1, 0.5)
    sim = Simulator(flow(), [bot], seed=0)
    sim.seed(10_000, 10_001)
    sim.warm_up(1e-9)
    mid = sim.book.mid()
    for lvl in list(sim.book.levels(Side.SELL)):
        for oid in list(lvl.orders):
            sim.book.cancel(oid)
    sim.run(0.0)
    assert sim.book.best_ask() is None and bot.arrival == mid


def test_trial_without_an_executor_is_a_session():
    makers = {"a": lambda: SymmetricQuoter(0.5)}
    res, cost = trial(4, makers, flow=flow)
    assert res == session(makers, 4, flow=flow, minutes=60.0)
    assert cost == 0.0


def test_trial_reports_the_executor_shortfall():
    res, cost = trial(0, {}, Executor(Side.BUY, [0.0], [5]), flow=flow, warm=0.0, minutes=0.0)
    assert res == {} and cost == pytest.approx(4.5)


def test_regime_lean_follows_the_flow():
    up, down = (RegimeLean(one_sided(1.0, 1.0, 0.02, 0.05, t=0.0), 2.0) for _ in range(2))
    assert up.filter.at(1.0)[1] == pytest.approx(up.filter.at(1.0)[2])
    for t in np.arange(0.2, 8.0, 0.2):
        up.observe(Print(t, Side.BUY, 10_001, 1))
        down.observe(Print(t, Side.SELL, 10_000, 1))
    p, q = up.filter.at(8.0), down.filter.at(8.0)
    assert p[1] > 0.95 and p[1] == pytest.approx(q[2])
    assert up.shift * (p[1] - p[2]) == pytest.approx(-down.shift * (q[1] - q[2]))


def test_filtered_maker_ends_on_the_executor_side():
    model = Model(1.0, 2.1, 3.6, 0.08, 5)
    makers = {
        "plain": lambda: GLFT(model, keep=1),
        "lean": lambda: GLFT(
            model, keep=1, signal=RegimeLean(one_sided(SKY.mu, 1.5, 1 / 60, 1 / 20), 1.0)
        ),
    }
    pos = {"plain": [], "lean": []}
    for seed in range(12):
        for name, make in makers.items():
            buyer = Executor(Side.BUY, np.arange(5.0, 35.0), np.arange(1, 31))
            res, _ = trial(seed, {name: make}, buyer, flow=flow, minutes=35.0)
            pos[name].append(res[name].position)
    assert np.mean(pos["lean"]) > np.mean(pos["plain"]) + 1.0


def test_scatter_splits_the_order():
    rng = np.random.default_rng(0)
    for n in (1, 4, 20):
        times, done = scatter(20, 10.0, n, rng)
        assert len(times) == len(done) == n
        assert np.all(np.diff(times) >= 0) and 0 <= times[0] and times[-1] < 10.0
        assert np.all(np.diff(done) > 0) and done[0] >= 1 and done[-1] == 20
    assert scatter(5, 1.0, 5, rng)[1].tolist() == [1, 2, 3, 4, 5]
    with pytest.raises(ValueError):
        scatter(5, 1.0, 6, rng)


class Tape(Agent):
    def __init__(self):
        self.prints = []

    def on_trade(self, ctx, p):
        self.prints.append(p)


def test_pov_keeps_its_share_of_the_volume():
    pov, tape = Pov(Side.BUY, 12, 0.3, start=5.0), Tape()
    sim = Simulator(flow(), [tape, pov], seed=2, record_every=None)
    sim.seed(10_000, 10_001)
    sim.run(60.0)
    fills = sim.contexts[1].account.fills
    assert sim.contexts[1].position == 12 and not any(f.maker for f in fills)
    assert min(f.t for f in fills) > 5.0
    assert pov.arrival == sim.tape.at(5.0)
    own = 0
    for f in fills:
        own += f.qty
        seen = sum(p.qty for p in tape.prints if 5.0 <= p.t <= f.t)
        assert own == min(12, math.floor(0.3 / 0.7 * (seen - own)))
    with pytest.raises(ValueError):
        Pov(Side.BUY, 1, 1.0)


def test_passive_rests_at_the_touch_and_finishes_at_the_deadline():
    times = np.arange(0.0, 21.0, 2.0)
    bot = Passive(Side.BUY, times, np.arange(0, 11) * 2, slack=20)
    sim = Simulator(flow(), [bot], seed=4, record_every=None)
    sim.seed(10_000, 10_001)
    sim.warm_up(30.0)
    ctx = sim.contexts[0]
    resting = 0
    for _ in range(46):
        sim.run(0.5)
        for o in ctx.orders.values():
            assert o.price == sim.book.best_bid() and o.price >= ctx.bid
            resting += 1
        if 1 <= bot._k <= 10:
            assert ctx.position + sum(o.qty for o in ctx.orders.values()) == 2 * bot._k
    assert resting > 10
    fills = ctx.account.fills
    assert ctx.position == 20 and not ctx.orders
    assert all(f.maker for f in fills if f.t < 50.0)
    assert any(f.maker for f in fills)
    assert min(f.t for f in fills if not f.maker) == 50.0


def test_passive_seller_crosses_what_it_is_behind():
    bot = Passive(Side.SELL, [0.0, 1e-6, 5.0], [0, 6, 6], slack=2)
    ctx = run(bot, 10.0)
    taken = sum(f.qty for f in ctx.account.fills if not f.maker and f.t < 1.0)
    assert taken == 4 and ctx.position == -6


def test_trial_takes_any_executor():
    _, cost = trial(1, {}, Pov(Side.SELL, 5, 0.5), flow=flow, warm=10.0, minutes=30.0)
    assert cost > 0.0
