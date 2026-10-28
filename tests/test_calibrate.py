import random

import numpy as np
import pytest

from mmsim.book import Book, EventType, Reason, Side
from mmsim.calibrate import Calibration, calibrate
from mmsim.flow import CSTParams, ZeroIntelligence
from mmsim.hawkes import Hawkes
from mmsim.itch import NS_PER_MINUTE, Msg

B, S = Side.BUY, Side.SELL


def sec(s, kind, ref, side=B, px=0, qty=0):
    return Msg(round(s * NS_PER_MINUTE / 60), kind, ref, side, px * 100, qty)


def test_hand_worked_feed():
    msgs = [
        sec(1, "A", 1, B, 99, 100),
        sec(2, "A", 2, S, 101, 200),
        sec(3, "A", 3, B, 98, 300),
        sec(90.5, "A", 4, S, 102, 50),
        sec(120.5, "X", 3, qty=100),
        sec(150.5, "E", 1, qty=100),
        sec(150.5, "E", 3, qty=50),
        sec(156.5, "P", 0, B, 99, 30),
        sec(168.5, "D", 4),
        sec(200, "A", 5, B, 97, 1000),
    ]
    c = calibrate(msgs, 1.0, 3.0, depth=5)
    assert c.T == 2.0 and list(c.limit_sizes) == [50] and c.hidden_shares == 30
    assert c.lam.tolist() == [[0, 0, 0, 0, 0], [0, 0, 0.5, 0, 0]]
    assert c.queue[B] == pytest.approx(np.array([0, 9100, 28650, 0, 0]) / 120)
    assert c.queue[S] == pytest.approx(np.array([0, 18200, 8800, 900, 0]) / 120)
    assert c.theta[B, 2] == pytest.approx(100 / 2 / (28650 / 120))
    assert c.theta[S, 3] == pytest.approx(50 / 2 / 7.5)
    assert np.count_nonzero(c.theta) == 2
    assert c.market_times == pytest.approx([90.5 / 60]) and c.market_sides.tolist() == [S]
    assert c.market_sizes.tolist() == [150] and c.cancel_shares == 75
    assert c.mu == pytest.approx(150 / 2 / 2 / 50)
    assert calibrate(msgs[:-1], 1.0, 3.0, depth=5).queue == pytest.approx(c.queue)


def zi_feed(params, minutes, seed):
    rng, ids = random.Random(seed), iter(range(1, 10**9))
    flow, book = ZeroIntelligence(params), Book()
    flow.seed(book, 100, 101, ids)
    msgs = [Msg(0, "A", o.id, o.side, o.price * 100, o.qty) for s in Side for o in book.orders(s)]
    t = 0.0
    while (step := flow.step(book, rng, minutes - t, ids)) is not None:
        t += step[0]
        ns = round(t * NS_PER_MINUTE)
        events = step[1](book)
        for e in events:
            if e.type == EventType.TRADE:
                msgs.append(Msg(ns, "E", e.maker, qty=e.qty))
            elif e.type == EventType.CANCELED and e.reason == Reason.USER:
                msgs.append(Msg(ns, "D", e.id))
        if events[0].type == EventType.ACCEPTED and e.id in book:
            msgs.append(Msg(ns, "A", e.id, e.side, e.price * 100, e.qty))
    return msgs


def test_recovers_the_zero_intelligence_model():
    truth = CSTParams(lam=(1.6, 1.2, 0.9, 0.7), theta=(0.6, 0.5, 0.4, 0.3), mu=0.9)
    c = calibrate(zi_feed(truth, 1500.0, 3), 100.0, 1500.0, depth=4)
    fit = c.cst()
    n = np.array(truth.lam) * 2 * c.T
    assert np.all(np.abs(np.array(fit.lam) / truth.lam - 1) < 4 / np.sqrt(n))
    assert fit.theta == pytest.approx(truth.theta, rel=0.08)
    assert fit.mu == pytest.approx(truth.mu, rel=4 / np.sqrt(truth.mu * 2 * c.T))
    assert c.limit_size == 1 and set(c.market_sizes) == {1.0}


def test_hawkes_comes_back_in_minutes():
    truth = Hawkes.make([20.0, 30.0], [[600.0, 120.0], [0.0, 900.0]], 2400.0)
    times, marks = truth.simulate(120.0, random.Random(7))
    empty = np.zeros(0)
    c = Calibration(120.0, empty, empty, empty, empty, 0.0, times, marks, empty, 0)
    h = c.hawkes()
    assert h.mu == pytest.approx(truth.mu, rel=0.15)
    assert h.branching == pytest.approx(truth.branching, abs=0.04)
    assert h.beta[1, 1] == pytest.approx(2400.0, rel=0.15)
