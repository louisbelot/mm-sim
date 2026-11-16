import numpy as np
import pytest

from mmsim import hidden as h
from mmsim import robust as r

SMALL = h.World(T=0.2, Q=6, phi=0.01, eta=0.05, rate=(1.0, 1.0))


def test_world_rejects_bad_parameters():
    with pytest.raises(ValueError):
        h.World(A=(250.0, 40.0))
    with pytest.raises(ValueError):
        h.World(k=(0.0, 1.0))
    with pytest.raises(ValueError):
        h.World(rate=(-1.0, 1.0))
    with pytest.raises(ValueError):
        h.World(pi0=2.0)


def test_update_is_bayes_on_fills_then_one_markov_step():
    w = h.World(rate=(3.0, 1.0))
    pi, db, da = 0.3, 0.4, 1.1
    a, k = np.array(w.A) * w.dt, np.array(w.k)
    pb, pa = a * np.exp(-k * db), a * np.exp(-k * da)
    for bought, sold in h.OUTCOMES:
        L = (pb if bought else 1 - pb) * (pa if sold else 1 - pa)
        post = pi * L[1] / (pi * L[1] + (1 - pi) * L[0])
        want = post * w.P[1, 1] + (1 - post) * w.P[0, 1]
        got = h.update(w, pi, db, da, bool(bought), bool(sold), False, False)
        assert got == pytest.approx(want, abs=1e-15)


def test_posterior_is_a_martingale():
    w = h.World()
    pi = np.linspace(0, 1, 11)
    total, mean = 0.0, 0.0
    for _, _, m, post in h.outcomes(w, pi, 0.3, 2.0, False, False):
        total, mean = total + m, mean + m * post
    assert np.allclose(total, 1.0, atol=1e-15)
    assert np.allclose(mean, pi, atol=1e-15)


def test_filter_drift_between_fills_matches_continuous_time():
    pi, db, da = 0.4, 0.5, 0.9
    errors = []
    for dt in (1e-3, 5e-4):
        w = h.World(dt=dt, rate=(2.0, 3.0))
        lam = [w.A[s] * (np.exp(-w.k[s] * db) + np.exp(-w.k[s] * da)) for s in (0, 1)]
        drift = -(lam[1] - lam[0]) * pi * (1 - pi) + w.rate[0] * (1 - pi) - w.rate[1] * pi
        got = h.update(w, pi, db, da, False, False, False, False) - pi
        errors.append(abs(got - drift * dt))
    assert errors[1] < 0.3 * errors[0]
    assert errors[0] < 1e-3


def test_fills_carry_no_information_at_the_crossing_depth():
    w = h.World()
    d = w.crossing
    assert np.isfinite(d)
    assert h.information(w, 0.5, d, d) == pytest.approx(0.0, abs=1e-15)
    assert h.information(w, 0.5, d / 2, d) > 0
    assert h.information(w, 0.5, 2 * d, d) > 0
    same = h.World(A=(100.0, 100.0), k=(1.0, 1.0))
    assert h.information(same, 0.3, 0.2, 0.7) == pytest.approx(0.0, abs=1e-15)


def test_interp_is_exact_on_nodes_lines_and_inner_quadratics():
    G = 11
    x = np.linspace(0, 1, G)
    W = np.stack([x**2, 3 * x - 1, np.ones(G)], axis=1)
    assert np.allclose(h.interp(W, x[:, None], np.arange(3)[None]), W, atol=1e-14)
    pi = np.linspace(0, 1, 37)
    assert np.allclose(h.interp(W, pi, 1), 3 * pi - 1, atol=1e-14)
    inner = pi[(pi > 0.1) & (pi < 0.9)]
    assert np.allclose(h.interp(W, inner, 0), inner**2, atol=1e-14)


def test_best_matches_a_brute_force_search():
    w = h.World(T=0.01, Q=3, phi=0.05)
    c = h.cells(w, 9)
    pi, q = np.linspace(0, 1, 9)[:, None], w.q[None, :]
    W = -0.05 * q**2 + 0.4 * (pi - 0.5) ** 2 + 0.1 * pi * q

    def f(b, a):
        return h.objective(w, W, c, b, a)

    bid, ask = h.best(w, f, c)
    got = f(bid, ask)
    d = np.linspace(0, 5 / min(w.k), 301)
    brute = np.max([f(np.full((9, 7), x), d[:, None, None]).max(axis=0) for x in d], axis=0)
    assert np.all(got >= brute - 1e-9)


def test_identical_regimes_reduce_to_the_one_regime_solution():
    w = h.World(T=0.2, Q=6, A=(140.0, 140.0), k=(1.5, 1.5), phi=0.01, eta=0.05)
    plan = h.bayes(w, 11)
    game = r.Game(T=w.T, dt=w.dt, sigma=w.sigma, Q=w.Q, zeta=w.phi * w.sigma**2 * w.dt, eta=w.eta)
    one = r.best_response(game, r.Market(0.0, 140.0, 1.5))
    assert np.allclose(plan.V, one.V[:, None], atol=1e-10)
    inner = np.isfinite(one.bid)
    assert np.allclose(plan.bid[:, :, inner[0]], one.bid[:, None, inner[0]], atol=1e-6)
