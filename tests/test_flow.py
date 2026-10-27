import itertools
import math
import random
from math import comb

import numpy as np
import pytest

from mmsim.book import Book, EventType, OrderType, Side
from mmsim.bots import GLFT
from mmsim.flow import (
    CSTParams,
    HawkesFlow,
    PairedFlow,
    ZeroIntelligence,
    prob_fill_first,
    prob_mid_up,
)
from mmsim.glft import Model
from mmsim.hawkes import Hawkes, fit
from mmsim.sim import Agent, Latency, Simulator

SKY = CSTParams.sky_perfect()
LAM1, MU, THETA1 = SKY.lam[0], SKY.mu, SKY.theta[0]

CST_TABLE3 = [
    [0.500, 0.336, 0.259, 0.216, 0.188],
    [0.664, 0.500, 0.407, 0.348, 0.307],
    [0.741, 0.593, 0.500, 0.437, 0.391],
    [0.784, 0.652, 0.563, 0.500, 0.452],
    [0.812, 0.693, 0.609, 0.548, 0.500],
]


def test_sky_perfect_params():
    assert SKY.lam[:5] == (1.85, 1.51, 1.09, 0.88, 0.77)
    assert SKY.lam[9] == pytest.approx(1.92 / 10**0.52)
    assert SKY.theta_at(5) == SKY.theta_at(40) == 0.47
    assert len(CSTParams.sky_perfect(depth=3).lam) == 3


def test_mid_up_matches_cst_table3():
    for b in range(1, 6):
        for a in range(1, 6):
            assert prob_mid_up(a, b, LAM1, MU, THETA1) == pytest.approx(
                CST_TABLE3[b - 1][a - 1], abs=1.5e-3
            )


def test_mid_up_is_antisymmetric():
    for a, b in [(1, 1), (1, 4), (3, 7), (12, 2)]:
        assert prob_mid_up(a, b, LAM1, MU, THETA1) + prob_mid_up(b, a, LAM1, MU, THETA1) == (
            pytest.approx(1, abs=1e-12)
        )


def race_of_market_orders(first, second):
    n = first + second - 1
    return sum(comb(n, k) for k in range(first, n + 1)) / 2**n


@pytest.mark.parametrize("a,b", [(1, 1), (1, 3), (4, 2), (5, 5)])
def test_pure_market_order_limit(a, b):
    assert prob_mid_up(a, b, 0.0, 1.0, 0.0) == pytest.approx(race_of_market_orders(a, b))
    assert prob_fill_first(a, b, 0.0, 1.0, 0.0) == pytest.approx(race_of_market_orders(b, a))


def test_truncation_does_not_matter():
    assert prob_mid_up(3, 2, LAM1, MU, THETA1, n_max=40) == pytest.approx(
        prob_mid_up(3, 2, LAM1, MU, THETA1, n_max=120), abs=1e-10
    )


def one_tick_book(a, b, ids):
    book = Book()
    for _ in range(b):
        book.submit(next(ids), Side.BUY, 100, 1)
    for _ in range(a):
        book.submit(next(ids), Side.SELL, 101, 1)
    for j in range(1, 10):
        for _ in range(2):
            book.submit(next(ids), Side.BUY, 100 - j, 1)
            book.submit(next(ids), Side.SELL, 101 + j, 1)
    return book


FLOWS = [ZeroIntelligence, PairedFlow]


@pytest.mark.parametrize("make", FLOWS)
@pytest.mark.parametrize("a,b", [(1, 1), (2, 1), (1, 3), (4, 2)])
def test_simulated_mid_moves_match_exact(a, b, make):
    rng, n, ups = random.Random(a * 10 + b), 1500, 0
    for _ in range(n):
        ids = itertools.count(1)
        book, flow = one_tick_book(a, b, ids), make(SKY)
        while (book.best_bid(), book.best_ask()) == (100, 101):
            flow.step(book, rng, math.inf, ids)[1](book)
        ups += book.mid() > 100.5
    p = prob_mid_up(a, b, LAM1, MU, THETA1)
    assert abs(ups / n - p) < 4 * math.sqrt(p * (1 - p) / n)


@pytest.mark.parametrize("make", FLOWS)
def test_long_run_rates_and_invariants(make):
    rng, ids, book, flow = random.Random(5), itertools.count(1), Book(), make(SKY)
    flow.seed(book, 1000, 1001, ids)
    t, horizon, limits, events = 0.0, 2000.0, 0, 0
    while True:
        step = flow.step(book, rng, horizon - t, ids)
        if step is None:
            break
        dt, fire = step
        t += dt
        ev = fire(book)
        events += 1
        if ev[0].type == EventType.ACCEPTED and ev[0].price != 0:
            limits += 1
        if events % 1000 == 0:
            assert book.audit() == ""
            assert all(o.qty == 1 for side in Side for o in book.orders(side))
    mean = 2 * sum(SKY.lam) * horizon
    assert abs(limits - mean) < 4 * math.sqrt(mean)


@pytest.mark.parametrize("make", FLOWS)
def test_flow_never_cancels_other_owners(make):
    rng, ids = random.Random(3), itertools.count(1)
    book, flow = one_tick_book(1, 1, ids), make(SKY)
    book.submit(10**6, Side.SELL, 300, 1, owner=7)
    for _ in range(3000):
        flow.step(book, rng, math.inf, ids)[1](book)
    assert 10**6 in book


@pytest.mark.parametrize("make", FLOWS)
def test_needs_two_sided_book(make):
    book = Book()
    book.submit(1, Side.BUY, 100, 1)
    with pytest.raises(ValueError):
        make(SKY).step(book, random.Random(0), 1.0, itertools.count(2))


@pytest.mark.parametrize("make", FLOWS)
def test_market_orders_take_one_lot(make):
    rng, ids = random.Random(11), itertools.count(1)
    book, flow = one_tick_book(3, 3, ids), make(SKY)
    seen = 0
    for _ in range(2000):
        ev = flow.step(book, rng, math.inf, ids)[1](book)
        if ev[0].type == EventType.ACCEPTED and ev[0].price == 0:
            seen += 1
            assert ev[0].qty == 1
            assert sum(e.qty for e in ev if e.type == EventType.TRADE) == 1
    assert seen > 0


@pytest.mark.parametrize("make", [HawkesFlow, PairedFlow])
def test_hawkes_flow_rate_and_refit(make):
    h = Hawkes.symmetric(0.94, 0.6, 0.1, 6.0)
    flow = make(SKY, h)
    book, ids, rng = Book(), itertools.count(1), random.Random(1)
    flow.seed(book, 1000, 1001, ids)
    T, t = 3000.0, 0.0
    times, marks = [], []
    while (step := flow.step(book, rng, T - t, ids)) is not None:
        t += step[0]
        ev = step[1](book)
        if ev[0].type == EventType.ACCEPTED and ev[0].price == 0:
            times.append(t)
            marks.append(int(ev[0].side))
    assert book.audit() == ""

    times, marks = np.array(times), np.array(marks)
    assert abs(len(times) / T - 1.88) < 4 * math.sqrt(1.88 / 0.3**2 / T)
    g = fit(times, marks, T, 2)
    assert np.allclose(g.branching, h.branching, atol=0.06)
    assert np.allclose(g.mu, h.mu, rtol=0.15)


def test_hawkes_flow_runs_under_the_simulator_with_latency():
    bot = GLFT(Model(1.0, 2.1, 3.6, 0.3, 5))
    flow = HawkesFlow(SKY, Hawkes.symmetric(0.94, 0.6, 0.1, 6.0))
    lat = {bot: Latency(feed=0.01, entry=0.01, response=0.01)}
    sim = Simulator(flow, [bot], seed=4, latency=lat)
    sim.seed(1000, 1001)
    sim.warm_up(10)
    sim.run(120)
    assert sim.book.audit() == ""
    assert len(sim.contexts[0].account.fills) > 10


@pytest.mark.parametrize("make", [HawkesFlow, PairedFlow])
def test_hawkes_flow_needs_two_dimensions(make):
    with pytest.raises(ValueError):
        make(SKY, Hawkes.make(1.0, 0.5, 1.0))


class Ticker(Agent):
    def on_start(self, ctx):
        ctx.set_timer(0.05)

    def on_timer(self, ctx):
        ctx.set_timer(0.05)


def session(flow, *agents):
    sim = Simulator(flow, agents, seed=7)
    sim.seed(1000, 1001)
    sim.warm_up(10)
    sim.run(60)
    return sim


def same_market(a, b):
    if len(a.prints) != len(b.prints):
        return False
    for p, q in zip(a.prints, b.prints, strict=True):
        if abs(p.t - q.t) > 1e-9 or p[1:] != q[1:]:
            return False
    resting = [[(o.id, o.price, o.qty) for s in Side for o in sim.book.orders(s)] for sim in (a, b)]
    return resting[0] == resting[1]


@pytest.mark.parametrize("hawkes", [None, Hawkes.symmetric(0.94, 0.6, 0.1, 6.0)])
def test_paired_flow_does_not_notice_timers(hawkes):
    quiet = session(PairedFlow(SKY, hawkes))
    busy = session(PairedFlow(SKY, hawkes), Ticker())
    assert len(quiet.prints) > 50
    assert same_market(quiet, busy)
    assert not same_market(session(ZeroIntelligence(SKY)), session(ZeroIntelligence(SKY), Ticker()))


def test_paired_flow_sends_the_same_market_orders_whatever_the_bots_do():
    hawkes = Hawkes.symmetric(0.94, 0.6, 0.1, 6.0)
    bots = [GLFT(Model(1.0, 2.1, 3.6, 0.3, 5), keep=k) for k in (0, 2)]
    runs = [session(PairedFlow(SKY, hawkes), bot) for bot in bots]
    a, b = (np.array([p.t for p in sim.prints]) for sim in runs)
    fills = [len(sim.contexts[0].account.fills) for sim in runs]
    assert fills[0] != fills[1] and min(fills) > 10
    near = np.abs(b[np.clip(np.searchsorted(b, a), 0, len(b) - 1)] - a) < 1e-9
    assert near.mean() > 0.99 and abs(len(a) - len(b)) <= len(a) // 100


class Sweeper(Agent):
    def __init__(self):
        self.times = []

    def on_start(self, ctx):
        ctx.set_timer(1.0)

    def on_timer(self, ctx):
        self.times.append(ctx.t)
        ctx.submit(Side.SELL, 0, 1000, kind=OrderType.MARKET)
        ctx.set_timer(1.0)


def test_paired_flow_market_orders_arrive_even_when_they_cannot_trade():
    hawkes = Hawkes.symmetric(0.94, 0.6, 0.1, 6.0)
    calm = session(PairedFlow(SKY, hawkes))
    swept = session(PairedFlow(SKY, hawkes), sweeper := Sweeper())
    a = np.array([p.t for p in calm.prints])
    b = np.array([p.t for p in swept.prints if p.t not in sweeper.times])
    assert len(sweeper.times) == 60 and len(b) < len(a)
    near = np.abs(a[np.clip(np.searchsorted(a, b - 1e-9), 0, len(a) - 1)] - b) < 1e-9
    assert near.mean() > 0.95
