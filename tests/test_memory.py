from dataclasses import replace

import numpy as np
import pytest

from mmsim import duopoly as du
from mmsim import learners as le
from mmsim import memory as mem

SMALL = du.Game(dt=0.01, A=40.0, k=1.5, Q=(3, 3), tick=0.2, reach=1.5, phi=0.5)
GAMMA = 0.95


def random_quotes(game, rng, lead=()):
    q1, q2 = game.grid()
    Q1, Q2 = game.Q
    d = du.depth(game, np.arange(game.J + 2))
    m = (*lead, *le.shape(game))
    cans = (q1 < Q1, q1 > -Q1, q2 < Q2, q2 > -Q2)
    return tuple(np.where(can, d[rng.integers(d.size, size=m)], np.inf) for can in cans)


def spread(quotes, lead=9):
    return tuple(np.broadcast_to(x, (lead, *x.shape)) for x in quotes)


def test_code_mirrors():
    rng = np.random.default_rng(0)
    qs = random_quotes(SMALL, rng, (9,))
    b1, a1, b2, a2 = qs
    c = mem.code(*qs)
    assert np.array_equal(mem.code(b2, a2, b1, a1), 8 - c)
    assert np.all(mem.code(b1, a1, b1, a1) == 4)
    assert set(np.unique(c)) <= set(range(9))


def test_chain_matches_memoryless_value():
    rng = np.random.default_rng(1)
    for _ in range(3):
        qs = random_quotes(SMALL, rng)
        P, _, _ = mem.chain(SMALL, tuple(x[None] for x in qs), 0)
        assert np.allclose(P.sum(axis=1), 1) and P.min() >= 0
        V1, V2 = mem.evaluate(SMALL, GAMMA, tuple(x[None] for x in qs), 0)
        W1, W2 = le.value(SMALL, GAMMA, qs)
        assert np.abs(V1[0] - W1).max() < 1e-10 and np.abs(V2[0] - W2).max() < 1e-10


def test_memory_that_is_ignored_changes_nothing():
    qs = spread(random_quotes(SMALL, np.random.default_rng(2)))
    V1, V2 = mem.evaluate(SMALL, GAMMA, qs, mem.code(*qs))
    W1, W2 = le.value(SMALL, GAMMA, tuple(x[0] for x in qs))
    assert np.abs(V1 - W1).max() < 1e-10 and np.abs(V2 - W2).max() < 1e-10


def test_improve_matches_enumeration():
    rng = np.random.default_rng(3)
    qs = random_quotes(SMALL, rng, (9,))
    V1, _ = mem.evaluate(SMALL, GAMMA, qs, mem.code(*qs))
    best, bid, ask = mem.improve(SMALL, GAMMA, V1, qs)
    q1, _ = SMALL.grid()
    Q1 = SMALL.Q[0]
    d = du.depth(SMALL, np.arange(SMALL.J + 2))
    shape = qs[0].shape
    brute = np.full(shape, -np.inf)
    for b in d:
        for a in d:
            b1 = np.broadcast_to(np.where(q1 < Q1, b, np.inf), shape)
            a1 = np.broadcast_to(np.where(q1 > -Q1, a, np.inf), shape)
            dev = (b1, a1, qs[2], qs[3])
            P, r1, _ = mem.chain(SMALL, dev, mem.code(*dev))
            brute = np.maximum(brute, (r1 + GAMMA * P @ V1.ravel()).reshape(shape))
    assert np.abs(best - brute).max() < 1e-12
    dev = (bid, ask, qs[2], qs[3])
    P, r1, _ = mem.chain(SMALL, dev, mem.code(*dev))
    assert np.abs((r1 + GAMMA * P @ V1.ravel()).reshape(shape) - best).max() < 1e-12


def test_nash_survives_memory():
    nash = le.markov(SMALL, GAMMA)
    qs = spread(nash.quotes)
    V1, V2 = mem.evaluate(SMALL, GAMMA, qs, mem.code(*qs))
    g1, g2 = mem.deviation(SMALL, GAMMA, qs, V1, V2)
    assert max(g1.max(), g2.max()) < 1e-9
    assert np.abs(mem.respond(SMALL, GAMMA, qs, 1) - V1).max() < 1e-8
    assert np.abs(mem.respond(SMALL, GAMMA, qs, 2) - V2).max() < 1e-8


@pytest.mark.parametrize("who", [1, 2])
def test_reply_to_memoryless_rival_is_memoryless(who):
    game = replace(SMALL, share=0.3, Q=(3, 2))
    qs = random_quotes(game, np.random.default_rng(4 + who))
    B = mem.respond(game, GAMMA, spread(qs), who)
    ref = le.respond(game, GAMMA, qs, who)[0]
    assert np.abs(B - ref).max() < 1e-8


def test_reply_beats_memory_profile():
    rng = np.random.default_rng(6)
    qs = random_quotes(SMALL, rng, (9,))
    V1, V2 = mem.evaluate(SMALL, GAMMA, qs, mem.code(*qs))
    for who, V, g in zip((1, 2), (V1, V2), mem.deviation(SMALL, GAMMA, qs, V1, V2), strict=True):
        B = mem.respond(SMALL, GAMMA, qs, who)
        assert np.all(B >= V - 1e-9)
        assert g.max() > 0 and (B - V).max() >= g.max() - 1e-9


def test_memory_learner_tables():
    a = le.train(SMALL, GAMMA, 2000, runs=2, window=500, seed=3, memory=True, sync=True, tau=100)
    b = le.train(SMALL, GAMMA, 2000, runs=2, window=500, seed=3, memory=True, sync=True, tau=100)
    assert np.array_equal(a.table1, b.table1)
    M = le.levels(SMALL).size
    assert a.table1.shape == (2, 7, 9, M * M)
    qs = mem.profile(SMALL, a.table1[0], a.table2[0])
    assert all(x.shape == (9, 7, 7) for x in qs)
    assert np.all(np.isinf(qs[0][:, -1])) and np.all(np.isinf(qs[3][:, :, 0]))


@pytest.mark.parametrize("memory", [False, True])
def test_decaying_rate_lone_learner(memory):
    game = replace(SMALL, Q=(3, 0))
    gamma = 0.9
    learned = le.train(game, gamma, 20_000, 4, 1.0, 3e-4, True, 5000, 1, memory, tau=20)
    opt = le.markov(game, gamma).V1[3, 0]
    for t1, t2 in zip(learned.table1, learned.table2, strict=True):
        qs = mem.profile(game, t1, t2) if memory else spread(le.quotes(game, t1, t2))
        V1, _ = mem.evaluate(game, gamma, qs, mem.code(*qs))
        assert V1[8, 3, 0] > 0.9 * opt
