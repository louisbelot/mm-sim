import numpy as np
import pytest

from mmsim import irl
from mmsim import robust as r

SMALL = r.Game(T=0.2, Q=12)
DELTAS = np.linspace(0, 2, 9)


def test_model_rejects_unknown_parameters():
    with pytest.raises(ValueError):
        irl.model(SMALL, ("gamma",), (0.1,))


def test_model_splits_game_and_market_parameters():
    g, m = irl.model(SMALL, ("zeta", "k"), (0.01, 1.8))
    assert g.zeta == 0.01 and g.nu == 0 and m.k == 1.8 and m.A == r.FIXED.A


def test_match_recovers_planted_penalties():
    true = r.best_response(r.Game(T=0.2, Q=12, zeta=0.004, nu=0.01, eta=0.3))
    visits = r.simulate(SMALL, true.bid, true.ask, paths=2000).visits
    fit = irl.match(SMALL, true.bid, true.ask, visits, ("zeta", "nu", "eta"))
    assert fit.params["zeta"] == pytest.approx(0.004, rel=1e-4)
    assert fit.params["nu"] == pytest.approx(0.01, rel=1e-4)
    assert fit.params["eta"] == pytest.approx(0.3, rel=1e-4)
    assert fit.loss < 1e-12


def test_match_recovers_the_market_a_bot_believes_in():
    true = r.best_response(r.Game(T=0.2, Q=12, zeta=0.005), r.Market(0.0, 120.0, 1.8))
    visits = r.simulate(SMALL, true.bid, true.ask, paths=2000).visits
    fit = irl.match(SMALL, true.bid, true.ask, visits, ("zeta", "A", "k"))
    assert fit.params["A"] == pytest.approx(120, rel=1e-3)
    assert fit.params["k"] == pytest.approx(1.8, rel=1e-4)


def test_drift_robust_bot_looks_like_a_linear_penalty():
    g = r.Game(T=0.3, Q=15)
    eq = r.equilibrium(g, "b")
    visits = r.simulate(g, eq.bid, eq.ask, paths=3000).visits
    both = irl.match(g, eq.bid, eq.ask, visits, ("zeta", "nu"))
    quad = irl.match(g, eq.bid, eq.ask, visits, ("zeta",))
    assert both.params["nu"] == pytest.approx(g.b[1] * g.dt, rel=0.15)
    assert both.params["zeta"] < 1e-4
    assert both.loss < quad.loss / 4


def test_qvalues_match_payoff_and_peak_at_the_best_response():
    sol = r.best_response(r.Game(T=0.2, Q=12, zeta=0.01))
    g = r.Game(T=0.2, Q=12, zeta=0.01)
    n, i = np.array([3, 3, 20]), np.array([12, 15, 9])
    Q = irl.qvalues(g, sol, r.FIXED, n, i, DELTAS)
    st = r.stage(g, sol.V[4]).at(np.array([15]))
    assert Q[1, 2 * 9 + 5] == pytest.approx(r.payoff(st, DELTAS[2], DELTAS[5], *r.FIXED)[0])
    assert np.all(Q.max(axis=1) <= sol.V[n, i] + 1e-12)


def test_loglik_is_a_normalized_softmax():
    q = np.array([[0.0, 1.0, 2.0], [5.0, 5.0, 5.0]])
    a = np.array([2, 0])
    p = np.exp(2.0 * q[0]) / np.exp(2.0 * q[0]).sum()
    assert irl.loglik(q, a, 2.0) == pytest.approx(np.log(p[2]) + np.log(1 / 3))


def test_boltzmann_recovers_penalty_and_temperature():
    g = r.Game(T=0.2, Q=12, zeta=0.02)
    sol = r.best_response(g)
    rng = np.random.default_rng(1)
    n = rng.integers(0, g.N, 6000)
    i = rng.integers(g.Q - 6, g.Q + 7, 6000)
    Q = irl.qvalues(g, sol, r.FIXED, n, i, DELTAS)
    z = 40.0 * Q
    p = np.exp(z - z.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    a = np.array([rng.choice(len(row), p=row) for row in p])
    fit = irl.boltzmann(SMALL, n, i, a, DELTAS, ("zeta",), beta0=10.0)
    assert fit.params["zeta"] == pytest.approx(0.02, rel=0.1)
    assert fit.params["beta"] == pytest.approx(40, rel=0.1)
