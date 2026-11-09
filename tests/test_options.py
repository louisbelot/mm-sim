import numpy as np
import pytest

from mmsim import multi
from mmsim import options as o


def chain():
    K, T = np.meshgrid([90.0, 100.0, 110.0], [0.1, 0.5])
    K, T = K.ravel(), T.ravel()
    return o.Chain.otm(100.0, K, T, 0.2 - 0.1 * np.log(K / 100))


def bbg(c, gamma=1e-3, eps=0.0):
    return o.market(
        c,
        lam=o.requests(c),
        alpha=0.7,
        beta=150.0,
        z=5e5 / c.g.price,
        eta=0.1,
        gamma=gamma,
        eps=eps,
    )


def test_greeks_finite_differences():
    S, K, T, s, h = 103.0, np.array([90.0, 100.0, 115.0]), 0.4, 0.25, 1e-4
    for call in (True, False):
        g = o.greeks(S, K, T, s, call)

        def p(S=S, T=T, s=s, call=call):
            return o.greeks(S, K, T, s, call).price

        assert np.allclose(g.delta, (p(S=S + h) - p(S=S - h)) / (2 * h), atol=1e-7)
        assert np.allclose(g.gamma, (p(S=S + h) - 2 * p() + p(S=S - h)) / h**2, atol=1e-4)
        assert np.allclose(g.vega, (p(s=s + h) - p(s=s - h)) / (2 * h), rtol=1e-6)
        assert np.allclose(g.theta, -(p(T=T + h) - p(T=T - h)) / (2 * h), rtol=1e-6)
        d = o.greeks(S, K, T, s + h, call).delta - o.greeks(S, K, T, s - h, call).delta
        assert np.allclose(g.vanna, d / (2 * h), atol=1e-6)
        v = o.greeks(S, K, T, s + h, call).vega - o.greeks(S, K, T, s - h, call).vega
        assert np.allclose(g.volga, v / (2 * h), atol=1e-5)
    c, p = o.greeks(S, K, T, s, True), o.greeks(S, K, T, s, False)
    assert np.allclose(c.price - p.price, S - K, atol=1e-12)


def test_greeks_match_vollab():
    bs = pytest.importorskip("vollab.bs")
    S, K, T, s = 98.0, np.array([80.0, 100.0, 120.0]), 0.75, 0.3
    for kind, call in (("call", True), ("put", False)):
        ours, ref = o.greeks(S, K, T, s, call), bs.greeks(S, K, T, 0.0, s, kind=kind)
        assert np.allclose(ours.price, bs.price(S, K, T, 0.0, s, kind=kind), atol=1e-12)
        for f in ("delta", "gamma", "vega", "theta", "vanna", "volga"):
            assert np.allclose(getattr(ours, f), getattr(ref, f), atol=1e-12)


def test_skew_depends_on_portfolio_vega_only():
    c = chain()
    m, V = bbg(c), c.g.vega
    q1 = np.zeros(c.d)
    q1[0] = 2e6 / V[0]
    q2 = np.zeros(c.d)
    q2[4] = 2e6 / V[4]
    for a, b in zip(m.quotes(q1, 0.004), m.quotes(q2, 0.004), strict=True):
        assert np.allclose(a, b, rtol=1e-10)


def test_vega_neutral_book_is_not_skewed():
    c = chain()
    m, V = bbg(c), c.g.vega
    q = np.zeros(c.d)
    q[1], q[3] = 1e6 / V[1], -1e6 / V[3]
    zero = m.quotes(np.zeros(c.d), 0.004)
    for a, b in zip(m.quotes(q, 0.004), zero, strict=True):
        assert np.allclose(a, b, rtol=1e-10)
    m = bbg(c, eps=0.05)
    assert not np.allclose(m.quotes(q, 0.004)[0], m.quotes(np.zeros(c.d), 0.004)[0])


def test_reduced_one_option_equals_exact():
    c = o.Chain.of(100.0, 100.0, 0.5, 0.2)
    m = bbg(c)
    zV, N = m.z[0] * c.g.vega[0], 6
    r = o.Reduced(m, c.g.vega, 0.1, N * zV, 0.004, points=2 * N + 1, rtol=1e-10)
    e = multi.Exact(m, (N,), 0.004, rtol=1e-10)
    assert np.allclose(r.w(0.004), e.theta(0.004), rtol=1e-7, atol=1e-3)
    n = np.arange(-N + 1, N)[:, None]
    vega = np.broadcast_to(c.g.vega, n.shape)
    for a, b in zip(r.policy(0.002, n, vega), e.policy(0.002, n), strict=True):
        assert np.allclose(a, b, rtol=1e-7)


def test_reduced_quotes_lean_with_vega():
    c = chain()
    m = bbg(c)
    r = o.Reduced(m, c.g.vega, 0.1, 3e7, 0.004, points=301)
    v = np.linspace(-1e7, 1e7, 11)
    bid, ask = r.quotes(0.004, v)
    assert np.all(np.diff(bid, axis=0) > 0)
    assert np.all(np.diff(ask, axis=0) < 0)
    assert np.allclose(bid, ask[::-1], rtol=1e-6)
    assert r.value() > 0


def test_closed_policy_rescales_by_current_vega():
    c = chain()
    m = bbg(c)
    n = np.array([[1.0, 0, -2, 0, 0, 3]])
    p = o.closed(m, c)
    for a, b in zip(p(0.004, n, c.g.vega[None]), m.policy(0.004, n), strict=True):
        assert np.allclose(a, b, rtol=1e-12)
    a, _ = p(0.004, n, 2 * c.g.vega[None])
    b, _ = m.policy(0.004, 2 * n)
    assert np.allclose(a, b, rtol=1e-12)
