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


def test_prior_counts_ties_by_share():
    game = replace(SMALL, share=1.0)
    M = le.levels(game).size
    first = le.prior(game, GAMMA, 1.0).reshape(M, M)
    second = le.prior(game, GAMMA, 0.0).reshape(M, M)
    assert np.all(first > second)
    assert np.allclose(first - first.T, 0)


def test_backup_mean_is_bellman():
    game = replace(SMALL, share=0.3)
    rng = np.random.default_rng(4)
    n = 400_000
    a = game.A * game.dt
    u = rng.random(n)
    buy, sell = u < a, (u >= a) & (u < 2 * a)
    r = rng.exponential(1 / game.k, n)
    mine = rng.random(n) < game.share
    d = le.levels(game)
    rival_b, rival_a = d[2], d[4]
    stay, up, down, cost = 1.0, 0.7, 1.4, 0.05
    for bid, ask in ((d[1], d[4]), (d[2], d[2]), (d[3], np.inf), (d[0], d[6])):
        x = le.backup(bid, ask, rival_b, rival_a, buy, sell, r, mine, stay, up, down, cost, GAMMA)
        p_b = a * np.exp(-game.k * bid) * ((bid < rival_b) + game.share * (bid == rival_b))
        p_a = a * np.exp(-game.k * ask) * ((ask < rival_a) + game.share * (ask == rival_a))
        gain_a = ask + GAMMA * down if np.isfinite(ask) else 0.0
        exact = p_b * (bid + GAMMA * up) + p_a * gain_a + (1 - p_b - p_a) * GAMMA * stay - cost
        assert abs(x.mean() - exact) < 4 * x.std() / np.sqrt(n) + 1e-12


@pytest.mark.parametrize(("sync", "alpha", "floor"), [(True, 0.01, 0.9), (False, 0.05, 0.85)])
def test_lone_learner_nears_best_response(sync, alpha, floor):
    game = replace(SMALL, Q=(3, 0))
    gamma = 0.9
    learned = le.train(game, gamma, 20_000, 4, alpha, 3e-4, sync, window=5000, seed=1)
    opt = le.markov(game, gamma).V1[3, 0]
    for t1, t2 in zip(learned.table1, learned.table2, strict=True):
        qs = le.quotes(game, t1, t2)
        assert np.all(np.isinf(qs[2])) and np.all(np.isinf(qs[3]))
        assert le.value(game, gamma, qs)[0][3, 0] > floor * opt


def test_train_is_seeded():
    a = le.train(SMALL, GAMMA, 2000, runs=2, window=500, seed=3)
    b = le.train(SMALL, GAMMA, 2000, runs=2, window=500, seed=3)
    assert np.array_equal(a.table1, b.table1) and np.array_equal(a.half, b.half)
    assert a.half.shape == (4, 2) and a.reward1.shape == (4, 2)
    assert a.table1.shape == (2, 7, le.levels(SMALL).size ** 2)
