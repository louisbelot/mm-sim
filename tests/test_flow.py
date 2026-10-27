import itertools
import math
import random
from math import comb

import numpy as np
import pytest

from mmsim.book import Book, EventType, Side
from mmsim.bots import GLFT
from mmsim.flow import CSTParams, HawkesFlow, ZeroIntelligence, prob_fill_first, prob_mid_up
from mmsim.glft import Model
from mmsim.hawkes import Hawkes, fit
from mmsim.sim import Latency, Simulator

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


@pytest.mark.parametrize("a,b", [(1, 1), (2, 1), (1, 3), (4, 2)])
def test_simulated_mid_moves_match_exact(a, b):
    rng, n, ups = random.Random(a * 10 + b), 1500, 0
    for _ in range(n):
        ids = itertools.count(1)
        book, flow = one_tick_book(a, b, ids), ZeroIntelligence(SKY)
        while (book.best_bid(), book.best_ask()) == (100, 101):
            flow.step(book, rng, math.inf, ids)[1](book)
        ups += book.mid() > 100.5
    p = prob_mid_up(a, b, LAM1, MU, THETA1)
    assert abs(ups / n - p) < 4 * math.sqrt(p * (1 - p) / n)


def test_long_run_rates_and_invariants():
    rng, ids, book, flow = random.Random(5), itertools.count(1), Book(), ZeroIntelligence(SKY)
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


def test_flow_never_cancels_other_owners():
    rng, ids = random.Random(3), itertools.count(1)
    book, flow = one_tick_book(1, 1, ids), ZeroIntelligence(SKY)
    book.submit(10**6, Side.SELL, 300, 1, owner=7)
    for _ in range(3000):
        flow.step(book, rng, math.inf, ids)[1](book)
    assert 10**6 in book


def test_needs_two_sided_book():
    book = Book()
    book.submit(1, Side.BUY, 100, 1)
    with pytest.raises(ValueError):
        ZeroIntelligence(SKY).step(book, random.Random(0), 1.0, itertools.count(2))


def test_market_orders_take_one_lot():
    rng, ids = random.Random(11), itertools.count(1)
    book, flow = one_tick_book(3, 3, ids), ZeroIntelligence(SKY)
    seen = 0
    for _ in range(2000):
        ev = flow.step(book, rng, math.inf, ids)[1](book)
        if ev[0].type == EventType.ACCEPTED and ev[0].price == 0:
            seen += 1
            assert ev[0].qty == 1
            assert sum(e.qty for e in ev if e.type == EventType.TRADE) == 1
    assert seen > 0


def test_hawkes_flow_rate_and_refit():
    h = Hawkes.symmetric(0.94, 0.6, 0.1, 6.0)
    flow = HawkesFlow(SKY, h)
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


def test_hawkes_flow_needs_two_dimensions():
    with pytest.raises(ValueError):
        HawkesFlow(SKY, Hawkes.make(1.0, 0.5, 1.0))
