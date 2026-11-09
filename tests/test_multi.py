import math

import numpy as np
import pytest
from scipy import sparse
from scipy.integrate import solve_ivp
from scipy.linalg import sqrtm
from scipy.optimize import minimize_scalar
from scipy.sparse.linalg import expm_multiply

from mmsim.glft import Model
from mmsim.multi import Exact, Exponential, Logistic, Market

RFQ = Logistic(30, 0.7, 30)
PAIR = Market.of([1.2, 0.6], 0.5, RFQ, 8e-6, z=1e4, mu=[0.1, -0.1])
SMALL = (4, 12)


def brute(f, p):
    res = minimize_scalar(
        lambda d: -f.rate(d) * (d - p),
        bounds=(p, p + 5),
        method="bounded",
        options={"xatol": 1e-12},
    )
    return res.x, -res.fun


@pytest.mark.parametrize("f", [RFQ, Logistic(5, -0.3, 4), Exponential(2.0, 1.5)])
@pytest.mark.parametrize("p", [-0.08, 0.0, 0.03, 0.2])
def test_hamiltonian_and_best_quote_match_brute_force(f, p):
    d, h = brute(f, p)
    assert f.delta(p) == pytest.approx(d, abs=1e-6)
    assert f.H(p) == pytest.approx(h, rel=1e-10)
    e = 1e-4
    fd = (f.H(p + e) - 2 * f.H(p) + f.H(p - e)) / e**2
    assert f.d2H(p) == pytest.approx(fd, rel=1e-5)


def test_riccati_closed_form_solves_the_odes():
    flow = (Exponential(3.0, 2.0), RFQ, Logistic(12, 0.2, 9))
    rho = np.array([[1, 0.4, -0.2], [0.4, 1, 0.7], [-0.2, 0.7, 1]])
    m = Market.of([0.8, 1.2, 0.5], rho, flow, 0.03, z=[1, 2, 0.5], mu=[0.05, -0.02, 0.1])
    D = np.diag(m.Dp)

    def rhs(_, y):
        A, B = y[:9].reshape(3, 3), y[9:]
        return np.concatenate(
            [(m.gamma / 2 * m.Sigma - 2 * A @ D @ A).ravel(), -m.mu - 2 * A @ D @ B]
        )

    taus = [0.05, 0.4, 3.0]
    sol = solve_ivp(
        rhs, (0, 3.0), np.zeros(12), t_eval=taus, method="Radau", rtol=1e-12, atol=1e-14
    )
    for j, tau in enumerate(taus):
        A, B = m.riccati(tau)
        assert np.allclose(A, sol.y[:9, j].reshape(3, 3), rtol=1e-7, atol=1e-12)
        assert np.allclose(B, sol.y[9:, j], rtol=1e-7, atol=1e-12)


def test_stationary_matrix_is_the_papers_gamma():
    A = PAIR.stationary()
    r = np.sqrt(PAIR.Dp)
    Gamma = sqrtm(np.outer(r, r) * PAIR.Sigma).real / np.outer(r, r)
    assert np.allclose(A, 0.5 * math.sqrt(PAIR.gamma) * Gamma, rtol=1e-10)
    assert np.allclose(2 * A @ np.diag(PAIR.Dp) @ A, PAIR.gamma / 2 * PAIR.Sigma, rtol=1e-10)


def test_stationary_drift_term_is_the_long_horizon_limit():
    A, B = PAIR.riccati(math.inf)
    assert np.allclose(2 * A @ np.diag(PAIR.Dp) @ B, -PAIR.mu, rtol=1e-10)
    assert np.allclose(B, PAIR.riccati(60.0)[1], rtol=1e-10)


def test_one_asset_is_the_glft_formula():
    g = Model(sigma=0.3, A=0.9, k=0.3, gamma=1e-7, Q=6)
    m = Market.of([0.3], 0.0, Exponential(0.9, 0.3), 1e-7)
    bid, ask = m.quotes(g.q[:, None])
    gb, ga = g.approx()
    assert np.allclose(bid[:, 0], gb, rtol=1e-5)
    assert np.allclose(ask[:, 0], ga, rtol=1e-5)


def test_uncorrelated_assets_quote_alone():
    m = Market.of([1.2, 0.6, 0.9], 0.0, (RFQ, Exponential(2.0, 1.5), RFQ), 1e-3, z=[2, 1, 3])
    A, B = m.riccati(5.0)
    assert np.allclose(A, np.diag(np.diag(A)), atol=0)
    for i in range(3):
        one = Market.of([m.Sigma[i, i] ** 0.5], 0.0, m.flow[i], m.gamma, z=m.z[i])
        assert A[i, i] == pytest.approx(one.riccati(5.0)[0][0, 0], rel=1e-12)


@pytest.mark.parametrize("rho", [-0.6, 0.0, 0.6])
def test_a_long_position_moves_the_other_book(rho):
    m = Market.of([1.2, 0.6], rho, RFQ, 8e-6, z=1e4)
    (b0, a0), (b1, a1) = m.quotes([0, 0]), m.quotes([1e4, 0])
    shift = np.sign(rho)
    assert np.sign(b1[1] - b0[1]) == shift and np.sign(a1[1] - a0[1]) == -shift
    assert b1[0] > b0[0] and a1[0] < a0[0]


def test_grid_matches_the_linearized_hjb():
    k = 1.3
    m = Market.of([0.7, 0.4], 0.6, (Exponential(2.0, k), Exponential(3.5, k)), 0.4, mu=[0.05, 0])
    ex = Exact(m, (3, 5), 2.0, rtol=1e-11)
    n = ex.nodes
    q = n * m.z
    run = q @ m.mu - m.gamma / 2 * np.einsum("pi,ij,pj->p", q, m.Sigma, q)
    L = sparse.diags(k * run).tolil()
    N = np.array(ex.N)
    index = {tuple(x): j for j, x in enumerate(n)}
    for j, x in enumerate(n):
        for i, f in enumerate(m.flow):
            for s in (1, -1):
                y = x.copy()
                y[i] += s
                if abs(y[i]) <= N[i]:
                    L[j, index[tuple(y)]] += f.A / math.e
    v = expm_multiply(ex.T * L.tocsr(), np.ones(len(n)))
    assert np.allclose(ex.theta(ex.T).ravel(), np.log(v) / k, atol=1e-8)


def test_uncorrelated_grid_separates():
    m = Market.of([1.2, 0.6], 0.0, RFQ, 8e-6, z=1e4, mu=[0.1, -0.1])
    th = Exact(m, SMALL, 2.0).theta(2.0)
    parts = [
        Exact(Market.of([s], 0.0, RFQ, 8e-6, z=1e4, mu=u), (n,), 2.0).theta(2.0)
        for s, u, n in zip([1.2, 0.6], [0.1, -0.1], SMALL, strict=True)
    ]
    assert np.allclose(th, parts[0][:, None] + parts[1][None, :], rtol=1e-7, atol=1e-3)


def test_evaluating_the_optimal_policy_gives_the_value():
    ex = Exact(PAIR, SMALL, 2.0)
    assert np.allclose(ex.evaluate(ex.policy), ex.theta(2.0), rtol=1e-6, atol=1e-2)


def test_optimal_beats_closed_form_beats_independent():
    c = SMALL
    ex = Exact(PAIR, SMALL, 2.0)
    best = ex.value()
    joint = ex.evaluate(PAIR.policy)[c]
    alone = ex.evaluate(PAIR.independent().policy)[c]
    assert best > joint > alone
    assert joint > best - 0.01 * abs(best)
    flat = Market.of([1.2, 0.6], 0.0, RFQ, 8e-6, z=1e4)
    ex0 = Exact(flat, SMALL, 2.0)
    assert ex0.evaluate(flat.policy)[c] == ex0.evaluate(flat.independent().policy)[c]


@pytest.mark.parametrize("rho", [-0.9, 0.0, 0.5, 0.95])
def test_target_inventory_is_the_markowitz_portfolio(rho):
    m = Market.of([1.2, 0.6], rho, RFQ, 8e-6, z=1e4, mu=[0.1, -0.1])
    assert np.allclose(m.target(), np.linalg.solve(m.Sigma, m.mu) / m.gamma, rtol=1e-9)
