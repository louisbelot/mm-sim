import random

import numpy as np
import pytest
from scripted import make, market

from mmsim.book import Side
from mmsim.bots import FlowLean, SymmetricQuoter
from mmsim.hawkes import Hawkes, Tracker
from mmsim.markout import Tape
from mmsim.sim import Print

SYM = Hawkes.symmetric(0.94, 0.6, 0.1, 6.0)


def quotes(ctx):
    return sorted((o.side, o.price) for o in ctx.orders.values())


def test_shift_leans_away_from_the_side_hit_and_decays():
    lean = FlowLean(SYM, 2.0)
    assert lean.shift(1.0) == 0
    lean.observe(Print(1.0, Side.SELL, 99, 1))
    assert lean.shift(1.0) == pytest.approx(-2.0)
    assert -2.0 < lean.shift(1.1) < lean.shift(1.2) < 0
    lean.observe(Print(1.2, Side.BUY, 101, 1))
    assert lean.shift(1.2) > -0.5


def test_fit_recovers_a_permanent_impact():
    times, marks = SYM.simulate(4000.0, random.Random(5))
    tape = Tape()
    tape.record(0.0, 10_000.0)
    net = np.cumsum(np.where(marks == 0, 1, -1))
    for t, n in zip(times, net, strict=True):
        tape.record(t, 10_000 + 0.4 * n)
    lean = FlowLean.fit(SYM, times, marks, tape, 1.0)
    assert lean.kappa == pytest.approx(0.4, rel=0.1)


def test_impact_is_the_move_each_order_makes():
    times, marks = SYM.simulate(4000.0, random.Random(5))
    rng = np.random.default_rng(5)
    sign = np.where(marks == 0, 1, -1)
    net = np.cumsum(sign)
    noise = np.cumsum(rng.choice([-0.5, 0.5], len(times)))
    tape = Tape()
    tape.record(0.0, 10_000.0)
    for t, n, s, e in zip(times, net, sign, noise, strict=True):
        tape.record(t - 1e-9, 10_000 + 0.4 * (n - s) + e)
        tape.record(t, 10_000 + 0.4 * n + e)
    lean = FlowLean.from_impact(SYM, times, marks, tape)
    assert lean.kappa == pytest.approx(0.4, rel=1e-9)

    lean.observe(Print(1.0, Side.SELL, 99, 1))
    tracker = Tracker(SYM)
    tracker.update(1.0, Side.SELL)
    e = tracker.excess(1.0)
    assert lean.shift(1.0) == pytest.approx(lean.kappa * (e[Side.BUY] - e[Side.SELL]))


def test_fit_and_impact_find_nothing_in_a_random_walk():
    times, marks = SYM.simulate(4000.0, random.Random(6))
    rng = np.random.default_rng(6)
    tape = Tape()
    tape.record(0.0, 10_000.0)
    for t, m in zip(times, 10_000 + np.cumsum(rng.choice([-0.5, 0.5], len(times))), strict=True):
        tape.record(t, m)
    assert abs(FlowLean.fit(SYM, times, marks, tape, 1.0).kappa) < 0.1
    assert abs(FlowLean.from_impact(SYM, times, marks, tape).kappa) < 4 * 0.5 / np.sqrt(len(times))


def test_quoter_moves_its_quotes_down_after_a_sell():
    plain = SymmetricQuoter()
    sim = make([(1.0, market(Side.SELL, 1))], plain)
    sim.run(1.5)
    assert quotes(sim.contexts[0]) == [(Side.BUY, 100), (Side.SELL, 101)]

    leaning = SymmetricQuoter(signal=FlowLean(SYM, 2.0))
    sim = make([(1.0, market(Side.SELL, 1))], leaning)
    sim.run(0.5)
    assert quotes(sim.contexts[0]) == [(Side.BUY, 100), (Side.SELL, 101)]
    sim.run(1.0)
    assert quotes(sim.contexts[0]) == [(Side.BUY, 98), (Side.SELL, 100)]
