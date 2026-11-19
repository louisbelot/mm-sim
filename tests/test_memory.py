import numpy as np

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
