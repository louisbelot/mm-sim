import dataclasses
import math
import random

import numpy as np
import pytest
from test_qr import feed, flat, flow_params

from mmsim import qr
from mmsim.book import Book
from mmsim.qr import CANCEL, KINDS, LIMIT, MARKET, SHARED, Clock, QueueReactive
from mmsim.sim import Simulator


def test_slot_clamps_at_both_ends():
    c = Clock((0.0, 10.0, 20.0, 30.0), ((1, 1, 1),) * 3)
    assert c.slot(-5.0) == (0, 15.0)
    assert c.slot(5.0) == (0, 5.0)
    assert c.slot(10.0) == (1, 10.0)
    assert c.slot(25.0) == (2, math.inf)
    assert c.slot(40.0) == (2, math.inf)


def synthetic(seed, X=20, H=5):
    rng = np.random.default_rng(seed)
    lam = rng.uniform(5, 50, (X, 3))
    s = rng.uniform(0.3, 3.0, (H, 3))
    s /= s.mean(axis=0)
    T = 100 * rng.dirichlet(np.ones(X), H).T
    n = rng.poisson(lam[:, None, :] * s[None] * T[..., None])
    f = qr.Fit(
        T.sum(),
        n.sum(axis=1)[None, :, None],
        T.sum(axis=1)[None, :, None],
        np.ones(1, dtype=np.int64),
        (1, 2),
        np.zeros((0, 6), dtype=np.int64),
        np.zeros((1, 1), dtype=np.int64),
        100.0 * np.arange(H + 1),
        T[None, :, None],
        n.sum(axis=0),
    )
    return f, s, T


def test_clock_recovers_the_multipliers():
    f, s, _ = synthetic(1)
    got = np.array(f.clock().scale)
    assert np.all(np.abs(got / s - 1) < 4 / np.sqrt(f.arrivals))


def test_clock_matches_arrivals_in_every_slot():
    f, _, T = synthetic(2)
    s = np.array(f.clock().scale)
    lam = f.counts.reshape(-1, 3) / (T @ s)
    assert s * (T.T @ lam) == pytest.approx(f.arrivals, rel=1e-8)
    assert np.diff(f.slots) @ s / f.T == pytest.approx(np.ones(3))


def test_shared_clock_is_one_multiplier():
    f, _, T = synthetic(3)
    s = np.array(f.clock(SHARED).scale)
    assert np.all(s == s[:, :1])
    lam = f.counts.reshape(-1, 3) / (T @ s)
    expected = (s * (T.T @ lam)).sum(axis=1)
    assert expected == pytest.approx(f.arrivals.sum(axis=1), rel=1e-8)


def test_flow_counts_follow_the_clock():
    clock = Clock((0.0, 50.0, 100.0, 150.0), ((0.5, 9, 9), (2.0, 9, 9), (1.0, 9, 9)))
    p = dataclasses.replace(flat(1, 4, lambda n: 6.0, lambda n: 0.0, lambda n: 0.0), clock=clock)
    book, rng, ids = Book(), random.Random(5), iter(range(1, 10**9))
    flow = QueueReactive(p, seed=5, start=-20.0)
    flow.seed(book, 100, 101, ids)
    t, times = -20.0, []
    while t < 150.0:
        step = flow.step(book, rng, min(0.37, 150.0 - t), ids)
        if step is None:
            t += min(0.37, 150.0 - t)
            continue
        t += step[0]
        times.append(t)
        step[1](book)
    assert flow.t == pytest.approx(t)
    got = np.histogram(times, [-20.0, 50.0, 100.0, 150.0])[0]
    want = np.array([12 * 70 * 0.5, 12 * 50 * 2.0, 12 * 50 * 1.0])
    assert np.all(np.abs(got - want) < 4 * np.sqrt(want))


def test_a_draw_does_not_jump_a_slot_edge():
    clock = Clock((0.0, 10.0, 20.0), ((0.01, 1, 1), (1.0, 1, 1)))
    p = dataclasses.replace(flat(1, 4, lambda n: 6.0, lambda n: 0.0, lambda n: 0.0), clock=clock)
    late = []
    for seed in range(300):
        book, ids = Book(), iter(range(1, 10**9))
        flow = QueueReactive(p, seed=seed)
        flow.seed(book, 100, 101, ids)
        dt = flow.step(book, random.Random(seed), 1e9, ids)[0]
        if dt > 10.0:
            late.append(dt - 10.0)
    assert len(late) > 50
    assert abs(np.mean(late) - 1 / 12) < 4 / 12 / np.sqrt(len(late))


def test_flat_clock_changes_nothing():
    def tape(p):
        sim = Simulator(QueueReactive(p, seed=3), seed=3)
        sim.seed(1000, 1001)
        sim.run(5.0)
        return sim.tape.t, sim.tape.mid

    p = flow_params()
    ones = dataclasses.replace(p, clock=Clock((0.0, 1.0), ((1.0, 1.0, 1.0),)))
    assert tape(p) == tape(ones)


def birth(n):
    return 30.0 / (1 + 0.3 * n) if n < 8 else 0.0


def cancel(n):
    return 6.0 + n


def market(n):
    return 9.0 / n


TRUTH = ((1.0, 1.0, 1.0), (0.5, 0.6, 0.4), (1.5, 1.3, 1.6))


@pytest.fixture(scope="module")
def msgs():
    p = dataclasses.replace(
        flat(2, 8, birth, cancel, market), clock=Clock((0.0, 1000.0, 2000.0, 3000.0), TRUTH)
    )
    return feed(p, 3000.0, 4)


def test_fit_recovers_the_clock_that_made_the_feed(msgs):
    f = qr.fit(msgs, 1000.0, 3000.0, depth=2, bins=8, slot=1000.0)
    s = np.array(f.clock().scale)
    se = np.sqrt(1 / f.arrivals[0] + 1 / f.arrivals[1])
    assert np.all(np.abs(np.log(s[1] / s[0]) - np.log(np.divide(*TRUTH[:0:-1]))) < 4 * se)
    got = f.params(regimes=False, min_events=1, clock=KINDS).rates[0][0]
    mean = np.mean(TRUTH[1:], axis=0)
    n_events = f.counts[0].sum(axis=0)
    for n in range(1, 7):
        for k, fn in ((LIMIT, birth), (CANCEL, cancel), (MARKET, market)):
            assert abs(got[n][k] / (fn(n) * mean[k]) - 1) < 4 / np.sqrt(n_events[n, k])


def test_rescale_reads_a_later_window_off_an_earlier_fit(msgs):
    early = qr.fit(msgs, 1000.0, 2000.0, depth=2, bins=8, slot=1000.0)
    late = qr.fit(msgs, 2000.0, 3000.0, depth=2, bins=8, aes=early.aes, edges=early.edges)
    p = early.params(regimes=False, min_events=1)
    got = np.array(qr.rescale(p, late).scale[0])
    want = np.divide(*TRUTH[:0:-1])
    se = np.sqrt(1 / early.arrivals[0] + 1 / late.arrivals[0])
    assert np.all(np.abs(np.log(got / want)) < 4 * se)
    shared = qr.rescale(p, late, SHARED).scale[0]
    assert shared[0] == shared[1] == shared[2]
    with pytest.raises(ValueError):
        qr.rescale(p, qr.fit(msgs, 2000.0, 3000.0, depth=2, bins=8))
