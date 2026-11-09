import numpy as np
import pytest

from mmsim import options as o


def chain():
    K, T = np.meshgrid([90.0, 100.0, 110.0], [0.1, 0.5])
    K, T = K.ravel(), T.ravel()
    return o.Chain.otm(100.0, K, T, 0.2 - 0.1 * np.log(K / 100))


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
