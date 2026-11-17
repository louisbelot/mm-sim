import numpy as np
import pytest

from mmsim import duopoly as du

SMALL = du.Game(T=1.0, dt=0.01, A=40.0, k=1.5, Q=(6, 6), tick=0.1, phi=0.5, reach=3.0)


def payoffs(game, g1, h1, g2, h2, can1, can2):
    none = game.J + 1
    j = np.arange(none + 1)
    d = du.depth(game, j)
    u1, u2 = du.side(game, d[:, None], d[None, :], g1, h1, g2, h2)
    u1 = np.where((j[:, None] < none) & ~np.asarray(can1), -np.inf, u1)
    u2 = np.where((j[None, :] < none) & ~np.asarray(can2), -np.inf, u2)
    return u1, u2


def equilibria(u1, u2):
    return (u1 >= u1.max(axis=0, keepdims=True)) & (u2 >= u2.max(axis=1, keepdims=True))


def random_stage(rng, shape):
    g1, g2 = rng.normal(0, 0.4, (2, *shape))
    h1 = g1 + rng.normal(0, 0.15, shape)
    h2 = g2 + rng.normal(0, 0.15, shape)
    can1, can2 = rng.random((2, *shape)) < 0.9
    return g1, h1, g2, h2, can1, can2


def test_game_checks():
    with pytest.raises(ValueError):
        du.Game(A=140.0, dt=0.005)
    with pytest.raises(ValueError):
        du.Game(share=1.5)
    with pytest.raises(ValueError):
        du.solve(du.Game(tick=0.0))


def test_peak_is_grid_argmax():
    game = du.Game(tick=0.07, reach=4.0)
    g = np.linspace(-3, 1.5, 301)
    d = du.depth(game, np.arange(game.J + 1))[:, None]
    brute = np.argmax(du.edge(game, d, g), axis=0)
    assert np.array_equal(du.peak(game, g), brute)


def test_stage_is_nash():
    game = du.Game(tick=0.1, reach=2.0)
    rng = np.random.default_rng(0)
    args = random_stage(rng, (40, 40))
    j1, j2, hard = du.stage(game, *args)
    missing = 0
    for i in np.ndindex(j1.shape):
        u1, u2 = payoffs(game, *(x[i] for x in args))
        if not equilibria(u1, u2).any():
            missing += 1
            continue
        assert u1[j1[i], j2[i]] >= u1[:, j2[i]].max() - 1e-15
        assert u2[j1[i], j2[i]] >= u2[j1[i], :].max() - 1e-15
    assert hard == missing
    assert np.mean((j1 <= game.J) | (j2 <= game.J)) > 0.5


def test_stage_selection_matches_exhaustive():
    game = du.Game(tick=0.1, reach=2.0)
    rng = np.random.default_rng(1)
    args = random_stage(rng, (25, 25))
    j1, j2, _ = du.stage(game, *args)
    for i in np.ndindex(j1.shape):
        a = [x[i] for x in args]
        u1, u2 = payoffs(game, *a)
        x1, x2 = du.exhaust(game, *a)
        assert (u1[x1, x2], u2[x1, x2]) == (u1[j1[i], j2[i]], u2[j1[i], j2[i]])


def test_symmetric_flat_stage_ties_one_tick_out():
    game = du.Game(tick=0.01)
    z = np.zeros(1)
    j1, j2, _ = du.stage(game, z, z, z, z, np.ones(1, bool), np.ones(1, bool))
    assert j1[0] == j2[0] == 1


def test_no_pure_equilibrium_falls_back_to_leader():
    game = du.Game(tick=0.05)
    g = [np.array([x]) for x in (0.5915, 0.6593, -0.6172, -0.6849)]
    yes = np.ones(1, bool)
    u1, u2 = payoffs(game, *(x[0] for x in g), True, True)
    assert not equilibria(u1, u2).any()
    j1, j2, hard = du.stage(game, *g, yes, yes)
    assert hard == 1
    best_follow = u2[j1[0]].max() == u2[j1[0], j2[0]] or u1[:, j2[0]].max() == u1[j1[0], j2[0]]
    assert best_follow


def test_one_maker_is_the_monopolist():
    game = du.Game(T=1.0, dt=0.01, A=40.0, Q=(8, 0), tick=0.05, phi=0.3, eta=0.1, beta=1.5)
    sol = du.solve(game)
    V, bid, ask = du.monopoly(game)
    np.testing.assert_allclose(sol.V1[:, :, 0], V, atol=1e-11)
    np.testing.assert_array_equal(sol.bid1[:, :, 0], bid)
    np.testing.assert_array_equal(sol.ask1[:, :, 0], ask)
    assert np.all(sol.V2 == 0)


def test_monopoly_matches_cartea_jaimungal_penalva():
    game = du.Game(T=1.0, dt=1e-4, A=10.0, k=1.5, Q=(4, 4), tick=0.0, phi=0.05, eta=0.02)
    V, _, _ = du.monopoly(game)
    q = np.arange(-4, 5)
    M = np.diag(-game.phi * game.k * q**2.0) + game.A / np.e * (np.eye(9, k=1) + np.eye(9, k=-1))
    w, U = np.linalg.eigh(M)
    tau = np.linspace(0, 1, 11)
    omega = np.einsum(
        "ij,tj,kj,k->ti", U, np.exp(np.outer(tau, w)), U, np.exp(-game.k * game.eta * q**2)
    )
    h = np.log(omega) / game.k
    assert (1 / game.k - np.diff(h, axis=1)).min() > 0
    np.testing.assert_allclose(V[0], h[-1], atol=2e-3)
    np.testing.assert_allclose(V[5000], h[5], atol=2e-3)


def test_symmetric_makers_mirror():
    sol = du.solve(SMALL)
    assert sol.hard == 0
    np.testing.assert_allclose(sol.V2, np.swapaxes(sol.V1, 1, 2), atol=1e-12)
    np.testing.assert_array_equal(sol.bid2, np.swapaxes(sol.bid1, 1, 2))
    np.testing.assert_array_equal(sol.ask2, np.swapaxes(sol.ask1, 1, 2))


def test_queue_priority_is_worth_something():
    Q = SMALL.Q[0]
    big = du.Game(**{**SMALL.__dict__, "tick": 0.5})
    split = du.solve(big)
    first = du.solve(du.Game(**{**big.__dict__, "share": 1.0}))
    assert first.V1[0, Q, Q] > split.V1[0, Q, Q] > first.V2[0, Q, Q]


def test_evaluate_reproduces_solve():
    game = du.Game(**{**SMALL.__dict__, "beta": 2.0})
    sol = du.solve(game)
    V1, V2 = du.evaluate(game, sol.quotes)
    np.testing.assert_allclose(V1, sol.V1, atol=1e-12)
    np.testing.assert_allclose(V2, sol.V2, atol=1e-12)
