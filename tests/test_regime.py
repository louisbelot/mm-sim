import numpy as np
import pytest
from scipy.stats import norm

from mmsim import regime as g
from mmsim import robust as r

SMALL = g.World(T=0.3, Q=15, phi=0.01, eta=0.05)


def test_world_rejects_bad_parameters():
    with pytest.raises(ValueError):
        g.World(A=250.0)
    with pytest.raises(ValueError):
        g.World(sigma=(0.0, 2.0))
    with pytest.raises(ValueError):
        g.World(rate=(-1.0, 1.0))
    with pytest.raises(ValueError):
        g.World(pi0=1.5)


def test_update_is_bayes_then_one_markov_step():
    w = g.World(b=(-3.0, 4.0), sigma=(1.0, 3.0), rate=(5.0, 1.0))
    rng = np.random.default_rng(0)
    pi, dz = rng.random(50), 0.3 * rng.standard_normal(50)
    l0 = norm.pdf(dz, w.b[0] * w.dt, w.sigma[0] * np.sqrt(w.dt))
    l1 = norm.pdf(dz, w.b[1] * w.dt, w.sigma[1] * np.sqrt(w.dt))
    post = np.stack([(1 - pi) * l0, pi * l1]) / ((1 - pi) * l0 + pi * l1)
    assert np.allclose(g.update(w, pi, dz), (w.P.T @ post)[1], atol=1e-14)


def test_certain_beliefs_stay_certain_without_switching():
    w = g.World(rate=(0.0, 0.0))
    dz = np.linspace(-1, 1, 7)
    assert np.all(g.update(w, 0.0, dz) == 0.0)
    assert np.all(g.update(w, 1.0, dz) == 1.0)


def test_informed_without_switching_is_two_best_responses():
    w = g.World(T=0.3, Q=15, phi=0.01, rate=(0.0, 0.0), sigma=(1.0, 3.0))
    full = g.informed(w)
    for j in (0, 1):
        s = r.best_response(w.game((w.b[j], w.b[j]), w.sigma[j] ** 2), r.Market(w.b[j], w.A, w.k))
        assert np.allclose(full.V[:, j], s.V, atol=1e-9)
        assert np.allclose(full.bid[:, j], s.bid, atol=1e-9)


def test_evaluating_the_informed_quotes_returns_their_value():
    full = g.informed(SMALL)
    assert np.allclose(g.evaluate(SMALL, full.bid, full.ask), full.V, atol=1e-9)


def test_uninformative_prices_reduce_bayes_to_a_fixed_drift():
    w = g.World(T=0.3, Q=15, phi=0.01, rate=(0.0, 0.0), sigma=(1e4, 1e4))
    plan, pi = g.bayes(w, 11)
    for i in (0, 3, 7, 10):
        b = (1 - pi[i]) * w.b[0] + pi[i] * w.b[1]
        s = r.best_response(w.game((b, b), 1e8), r.Market(b, w.A, w.k))
        assert np.allclose(plan.V[0, i], s.V[0], atol=1e-6)


def test_certain_bayes_matches_informed():
    w = g.World(T=0.3, Q=15, phi=0.01, rate=(0.0, 0.0))
    plan, _ = g.bayes(w, 21)
    full = g.informed(w)
    assert np.allclose(plan.V[:, 0], full.V[:, 0], atol=1e-9)
    assert np.allclose(plan.V[:, -1], full.V[:, 1], atol=1e-9)


def test_bayes_value_is_convex_in_the_belief():
    plan, _ = g.bayes(SMALL, 51)
    for n in (0, SMALL.N // 2, SMALL.N - 1):
        assert np.diff(plan.V[n], 2, axis=0).min() > -1e-9
