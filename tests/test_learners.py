from dataclasses import replace

import numpy as np
import pytest

from mmsim import duopoly as du
from mmsim import learners as le

SMALL = du.Game(dt=0.01, A=40.0, k=1.5, Q=(3, 3), tick=0.2, reach=1.5, phi=0.5)
GAMMA = 0.95


def iterated(game, gamma, quotes):
    V1 = V2 = np.zeros(le.shape(game))
    for _ in range(1000):
        V1, V2 = du.advance(game, gamma * V1, gamma * V2, quotes)
    return V1, V2


def random_quotes(game, rng):
    d = du.depth(game, np.arange(game.J + 2))
    return tuple(d[rng.integers(d.size, size=le.shape(game))] for _ in range(4))


def test_value_solves_bellman():
    rng = np.random.default_rng(1)
    for _ in range(3):
        quotes = random_quotes(SMALL, rng)
        V1, V2 = le.value(SMALL, GAMMA, quotes)
        W1, W2 = iterated(SMALL, GAMMA, quotes)
        assert np.abs(V1 - W1).max() < 1e-10
        assert np.abs(V2 - W2).max() < 1e-10


def test_markov_is_nash():
    nash = le.markov(SMALL, GAMMA)
    assert nash.eps == 0
    for who, V in ((1, nash.V1), (2, nash.V2)):
        B = le.respond(SMALL, GAMMA, nash.quotes, who)[0]
        assert np.abs(B - V).max() < 1e-8
    assert np.abs(nash.V1 - nash.V2.T).max() < 1e-10
    gaps = le.deviation(SMALL, GAMMA * nash.V1, GAMMA * nash.V2, nash.quotes)
    assert max(g.max() for g in gaps) <= 1e-10


def test_markov_falls_back_to_nearest():
    game = replace(SMALL, share=1.0, tick=0.1)
    nash = le.markov(game, GAMMA)
    assert nash.eps > 0
    assert nash.eps == pytest.approx(le.exploit(game, GAMMA, nash.quotes, nash.V1, nash.V2))
    V1, V2 = le.value(game, GAMMA, nash.quotes)
    assert np.abs(V1 - nash.V1).max() < 1e-12 and np.abs(V2 - nash.V2).max() < 1e-12
    assert nash.eps < 0.05 * nash.V1[3, 3]


def test_deviation_finds_profitable_moves():
    nash = le.markov(SMALL, GAMMA)
    Q = SMALL.Q[0]
    b1 = nash.quotes[0].copy()
    b1[Q, Q] += SMALL.tick
    quotes = (b1, *nash.quotes[1:])
    V1, V2 = le.value(SMALL, GAMMA, quotes)
    gb1 = le.deviation(SMALL, GAMMA * V1, GAMMA * V2, quotes)[0]
    assert gb1[Q, Q] > 0
    assert le.respond(SMALL, GAMMA, quotes, 1)[0][Q, Q] > V1[Q, Q]


def test_lone_maker_is_the_cartel():
    game = replace(SMALL, Q=(3, 0))
    nash = le.markov(game, GAMMA)
    cart = le.collusive(game, GAMMA)
    assert np.abs(nash.V1 - (cart.V1 + cart.V2)).max() < 1e-9
    assert np.all(np.isinf(nash.quotes[2])) and np.all(np.isinf(nash.quotes[3]))
    V, bid, ask = le.respond(game, GAMMA, nash.quotes, 1)
    assert np.abs(V - nash.V1).max() < 1e-8


def test_cartel_beats_nash_and_random_play():
    nash = le.markov(SMALL, GAMMA)
    cart = le.collusive(SMALL, GAMMA)
    total = cart.V1 + cart.V2
    assert np.all(total >= nash.V1 + nash.V2 - 1e-10)
    rng = np.random.default_rng(2)
    for _ in range(5):
        V1, V2 = le.value(SMALL, GAMMA, random_quotes(SMALL, rng))
        assert np.all(total >= V1 + V2 - 1e-10)
    Q = SMALL.Q[0]
    assert total[Q, Q] > 1.5 * (nash.V1 + nash.V2)[Q, Q]


def test_markov_needs_a_tick():
    with pytest.raises(ValueError):
        le.markov(replace(SMALL, tick=0.0), GAMMA)
