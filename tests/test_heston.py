import math

import numpy as np
import pytest

from mmsim import heston as h
from mmsim import options as o
from mmsim.multi import Logistic, Market

Q = h.Heston(3.0, 0.0225, 0.2, -0.5)
P = h.Heston(2.0, 0.04, 0.2, -0.5)

PAPER = [
    (8, 1.0, 2.06, 0.41, 0.1622),
    (8, 1.5, 2.11, 0.46, 0.1583),
    (8, 2.0, 2.16, 0.48, 0.1563),
    (8, 3.0, 2.27, 0.46, 0.1544),
    (9, 1.0, 1.21, 0.91, 0.1538),
    (9, 1.5, 1.31, 0.84, 0.1518),
    (9, 2.0, 1.41, 0.76, 0.1515),
    (9, 3.0, 1.57, 0.65, 0.1508),
    (10, 1.0, 0.58, 1.25, 0.1460),
    (10, 1.5, 0.71, 1.05, 0.1459),
    (10, 2.0, 0.83, 0.92, 0.1468),
    (10, 3.0, 1.02, 0.76, 0.1476),
    (11, 1.0, 0.21, 1.05, 0.1392),
    (11, 1.5, 0.33, 0.96, 0.1409),
    (11, 2.0, 0.44, 0.88, 0.1427),
    (11, 3.0, 0.62, 0.75, 0.1448),
    (12, 1.0, 0.06, 0.54, 0.1344),
    (12, 1.5, 0.13, 0.65, 0.1367),
    (12, 2.0, 0.21, 0.67, 0.1395),
    (12, 3.0, 0.36, 0.65, 0.1423),
]


def book(K=(9.0, 10.0, 11.0), T=(1.0, 2.0), gamma=1e-3, lam=252 * 30):
    K, T = (a.ravel() for a in np.meshgrid(K, T))
    pr = h.Pricer.of(Q, K, nu_lo=0.0144)
    g = pr.greeks(10.0, 0.0225, T)
    V = pr.vega(10.0, 0.0225, T)
    lam = lam / (1 + 0.7 * np.abs(10 - K))
    flow = tuple(Logistic(a, 0.7, 150 / v) for a, v in zip(lam, V, strict=True))
    m = Market(0.01 * np.outer(V, V), flow, gamma, 5e5 / g.price, np.zeros(len(K)))
    return pr, T, V, m


def test_fang_oosterlee_reference():
    m = h.Heston(1.5768, 0.0398, 0.5751, -0.5711)
    p = h.Pricer.of(m, [100.0], nu_lo=0.0175)
    assert p.greeks(100.0, 0.0175, 1.0).price[0] == pytest.approx(5.785155450, abs=5e-8)


def test_black_scholes_limit():
    K, T = np.array([80.0, 100.0, 125.0]), np.array([0.1, 0.5, 2.0])
    for call in (True, False):
        p = h.Pricer.of(h.Heston(2.0, 0.04, 1e-3, 0.0), K, call, nu_lo=0.04)
        ref = o.greeks(100.0, K, T, 0.2, call)
        g = p.greeks(100.0, 0.04, T)
        assert np.allclose(g.price, ref.price, atol=1e-7)
        assert np.allclose(g.delta, ref.delta, atol=1e-7)


def test_greeks_finite_differences():
    K, T, S, nu, e = np.array([8.0, 10.0, 12.0]), np.array([0.5, 1.0, 3.0]), 10.3, 0.03, 1e-5
    for call in (True, False):
        p = h.Pricer.of(Q, K, call, nu_lo=0.01)
        g = p.greeks(S, nu, T)
        up, dn = p.greeks(S + e, nu, T).price, p.greeks(S - e, nu, T).price
        assert np.allclose(g.delta, (up - dn) / (2 * e), atol=1e-7)
        up, dn = p.greeks(S, nu + e, T).price, p.greeks(S, nu - e, T).price
        assert np.allclose(g.dnu, (up - dn) / (2 * e), rtol=1e-6)
    c = h.Pricer.of(Q, K, True, nu_lo=0.01).greeks(S, nu, T)
    p = h.Pricer.of(Q, K, False, nu_lo=0.01).greeks(S, nu, T)
    assert np.allclose(c.price - p.price, S - K, atol=1e-12)


def test_characteristic_function_is_a_martingale():
    for T in (0.1, 1.0, 5.0):
        assert P.cf(0.0, T, 0.03) == pytest.approx(1.0, abs=1e-14)
        assert P.cf(-1j, T, 0.03) == pytest.approx(1.0, abs=1e-12)


def test_paper_chain():
    K = np.array([r[0] for r in PAPER], float)
    T = np.array([r[1] for r in PAPER])
    pr = h.Pricer.of(Q, K, nu_lo=0.0144)
    g, V = pr.greeks(10.0, 0.0225, T), pr.vega(10.0, 0.0225, T)
    vols = [h.iv(c, 10.0, k, t) for c, k, t in zip(g.price, K, T, strict=True)]
    assert np.abs(g.price - [r[2] for r in PAPER]).max() < 0.01
    assert np.abs(V - [r[3] for r in PAPER]).max() < 0.025
    assert np.abs(np.array(vols) - [r[4] for r in PAPER]).max() < 0.003


def test_same_drift_matches_one_dimensional_hjb():
    pr, T, V, m = book()
    one = o.Reduced(m, V, 0.1, 1e7, 0.0012, points=81)
    two = h.Reduced(m, V, Q, Q, 0.0144, 0.0324, 1e7, 0.0012, rows=5, points=81)
    W = two.w(0.0012)
    assert np.abs(W - one.w(0.0012)).max() < 1e-6 * np.abs(W).max()
    b1, a1 = one.quotes(0.0012, 2e6)
    b2, a2 = two.quotes(0.0012, [0.015, 0.03], 2e6)
    assert np.allclose(b2, b1, rtol=1e-8) and np.allclose(a2, a1, rtol=1e-8)


def test_variance_drift_tilts_toward_long_vega():
    pr, T, V, m = book(gamma=1e-5)
    flat = o.Reduced(m, V, 0.1, 1e7, 0.0012, points=81)
    up = h.Reduced(m, V, P, Q, 0.0144, 0.0324, 1e7, 0.0012, rows=5, points=81)
    dn = h.Reduced(m, V, Q, P, 0.0144, 0.0324, 1e7, 0.0012, rows=5, points=81)
    b0, a0 = flat.quotes(0.0012, 0.0)
    bu, au = up.quotes(0.0012, 0.0225, 0.0)
    bd, ad = dn.quotes(0.0012, 0.0225, 0.0)
    assert np.all(bu < b0) and np.all(au > a0)
    assert np.all(bd > b0) and np.all(ad < a0)
    assert up.value(0.0225) > flat.value()


def test_value_matches_its_own_model():
    pr, T, V, m = book()
    r = h.Reduced(m, V, P, Q, 0.0144, 0.0324, 1e7, 0.0012, rows=7, points=101)
    run = dict(nu0=0.0225, dt=0.0012 / 120, paths=3000, seed=3)
    x = h.own(r, r.quotes, **run)
    se = x.std() / math.sqrt(len(x))
    assert abs(x.mean() - r.value(0.0225)) < 4 * se
    lazy = h.own(r, lambda tau, nu, v: r.quotes(r.T, 0.0225, 0.0), **run)
    assert lazy.mean() < x.mean()
