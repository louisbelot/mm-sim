import numpy as np
import pytest

from mmsim import duopoly as du
from mmsim import learners as le
from mmsim import memory as mem
from mmsim import trigger as tr

SMALL = du.Game(dt=0.01, A=40.0, k=1.5, Q=(3, 3), tick=0.2, reach=1.5, phi=0.5)
GAMMA = 0.9


@pytest.fixture(scope="module")
def nash():
    return le.markov(SMALL, GAMMA)


@pytest.fixture(scope="module")
def cartel():
    return le.collusive(SMALL, GAMMA)


def test_phases_count_down():
    qs = tuple(np.zeros((2, 2)) for _ in range(4))
    _, nxt = tr.phases(qs, qs, 3)
    assert nxt[:, 0, 0].tolist() == [0, 2, 3, 0]
    assert tr.phases(qs, qs, None)[1][:, 0, 0].tolist() == [0, 1]
    with pytest.raises(ValueError):
        tr.phases(qs, qs, 0)


@pytest.mark.parametrize("length", [1, 4, None])
def test_nash_reverting_to_nash_is_nash(nash, length):
    res = tr.check(SMALL, GAMMA, nash.quotes, nash.quotes, length)
    assert res.gain < 1e-9
    assert np.abs(res.V1 - nash.V1).max() < 1e-9


@pytest.mark.parametrize("length", [1, 3, None])
def test_values_match_the_phase_chain(nash, cartel, length):
    quotes, nxt = tr.phases(cartel.quotes, nash.quotes, length)
    V1, V2 = mem.evaluate(SMALL, GAMMA, quotes, nxt)
    W1, W2 = tr.values(SMALL, GAMMA, cartel.quotes, nash.quotes, length)
    assert np.abs(V1 - W1).max() < 1e-10 and np.abs(V2 - W2).max() < 1e-10


def test_cheat_matches_enumeration(nash, cartel):
    quotes, nxt = tr.phases(cartel.quotes, nash.quotes, 2)
    V1, _ = mem.evaluate(SMALL, GAMMA, quotes, nxt)
    gain = tr.cheat(SMALL, GAMMA, V1, quotes)
    q1, _ = SMALL.grid()
    Q1 = SMALL.Q[0]
    shape = quotes[0].shape
    d = du.depth(SMALL, np.arange(SMALL.J + 2))
    brute = np.full(shape, -np.inf)
    for b in d:
        for a in d:
            b1 = np.broadcast_to(np.where(q1 < Q1, b, np.inf), shape)
            a1 = np.broadcast_to(np.where(q1 > -Q1, a, np.inf), shape)
            P, r1, _ = mem.chain(SMALL, (b1, a1, quotes[2], quotes[3]), 1)
            v = (r1 + GAMMA * P @ V1.ravel()).reshape(shape)
            on = (b1 == quotes[0]) & (a1 == quotes[1])
            brute = np.maximum(brute, np.where(on, -np.inf, v))
    assert np.abs(gain - (brute - V1)).max() < 1e-12


def test_grim_holds_a_markup_but_not_the_cartel(nash, cartel):
    assert tr.check(SMALL, GAMMA, cartel.quotes, nash.quotes, None).gain > 0.1
    soft = tr.check(SMALL, GAMMA, tr.widen(SMALL, nash.quotes, 1), nash.quotes, None)
    assert soft.gain < 1e-9
    Q = SMALL.Q[0]
    assert soft.V1[0, Q, Q] > nash.V1[Q, Q] + 0.01


def test_price_war_is_not_credible(nash, cartel):
    res = tr.check(SMALL, GAMMA, tr.widen(SMALL, nash.quotes, 1), tr.war(SMALL), None)
    assert res.gain1[0].max() < 0 and res.gain1[1].max() > 0


def test_widen_moves_by_ticks(nash):
    b = nash.quotes[0]
    w = tr.widen(SMALL, nash.quotes, 2)[0]
    ok = np.isfinite(b) & (b < du.depth(SMALL, SMALL.J - 2))
    assert np.allclose(w[ok] - b[ok], 2 * SMALL.tick)
    assert np.array_equal(np.isinf(w), np.isinf(b))
    assert w.max(initial=0, where=np.isfinite(w)) <= du.depth(SMALL, SMALL.J) + 1e-12


def test_shared_cartel_splits_evenly(cartel):
    qs = tr.shared(SMALL, cartel.quotes)
    V1, V2 = le.value(SMALL, GAMMA, qs)
    assert np.abs(V1 - V2.T).max() < 1e-10
    assert np.all(np.isinf(qs[0][-1])) and np.all(np.isinf(qs[3][:, 0]))
    Q = SMALL.Q[0]
    total = (cartel.V1 + cartel.V2)[Q, Q]
    assert 0.8 * total < 2 * V1[Q, Q] <= total + 1e-10
    assert abs(cartel.V1[Q, Q] - cartel.V2[Q, Q]) > abs(V1[Q, Q] - V2[Q, Q]) + 0.1
