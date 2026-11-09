import math

import numpy as np
import pytest

from mmsim import fees, lvr
from mmsim import venues as vn

FAST = vn.Market(P0=1.0, sigma=1.0, lam=50.0, noise=0.0, hours=10.0)


def se(a):
    return np.std(a, ddof=1) / math.sqrt(len(a))


def test_rules():
    s = np.array([0.5, 1.0, 2.0])
    assert fees.Static(0.003)(s).tolist() == [0.003] * 3
    assert fees.Threshold(0.001, 0.004, 1.0)(s).tolist() == [0.001, 0.001, 0.004]
    assert fees.Power(0.001, 0.002, 2, cap=0.006)(s) == pytest.approx([0.0015, 0.003, 0.006])


def test_regime_moments():
    r = fees.Regimes(spike=3.0, up=1.0, down=3.0)
    assert r.share == 0.25
    assert r.rms(2.0) == pytest.approx(2 * math.sqrt(0.75 + 0.25 * 9))


def test_ewma_one_step():
    e = fees.Ewma(np.array([0.2]), half_life=2.0)
    H = 2 / math.log(2)
    e.update(np.array([0.1]), 1.0)
    assert e.v == pytest.approx(0.04 * H * math.exp(-1 / H) + 0.01)
    e.update(np.array([0.0]), 2.0)
    assert e.sigma == pytest.approx(np.sqrt((0.04 * H * math.exp(-1 / H) + 0.01) / H / 2))


def test_ewma_is_unbiased_for_poisson_returns():
    rng = np.random.default_rng(0)
    sigma, n = 0.3, 4000
    e = fees.Ewma(np.full(n, 1.0), half_life=0.5)
    for _ in range(3000):
        tau = rng.exponential(1 / 400, n)
        e.update(sigma * np.sqrt(tau) * rng.standard_normal(n), tau)
    assert np.mean(e.sigma**2) == pytest.approx(sigma**2, rel=0.01)


def test_unknown_source():
    with pytest.raises(ValueError):
        vn.run(FAST, math.inf, 1.0, 0.01, source="oracle", paths=2)


def test_static_rule_is_the_float_fee():
    a = vn.run(FAST, 0.5, 1.0, 0.02, paths=20, seed=1)
    b = vn.run(FAST, 0.5, 1.0, fees.Static(0.02), paths=20, seed=1)
    assert np.array_equal(a.lp.total, b.lp.total)
    assert a.fee == pytest.approx(0.02)


def test_time_in_spike_from_a_calm_start():
    r = fees.Regimes(spike=2.0, up=0.5, down=1.5)
    s = vn.run(FAST, math.inf, 1.0, 0.05, regimes=r, paths=2000, seed=2)
    k = (r.up + r.down) * FAST.hours
    want = r.share * (1 - (1 - math.exp(-k)) / k)
    assert abs(s.spike.mean() - want) < 4 * se(s.spike)


def test_oracle_fee_is_quasi_static_in_each_regime():
    r = fees.Regimes(spike=3.0, up=0.5, down=1.0)
    rule = fees.Threshold(0.05, 0.15, 2.0)
    s = vn.run(FAST, math.inf, 1.0, rule, regimes=r, source="true", paths=1000, seed=5)
    calm, hot = s.lvr - s.lvr_spike, s.lvr_spike

    def ratio(f, sigma, g):
        return f(sigma, FAST.lam, g) / lvr.lvr_rate(sigma)

    arb = -s.lp.arb - s.lp.arb_fees
    want = calm * ratio(lvr.arb_rate, 1.0, 0.05) + hot * ratio(lvr.arb_rate, 3.0, 0.15)
    assert abs(arb.mean() - want.mean()) < 4 * se(arb - want)
    fee = s.lp.arb_fees
    want = calm * ratio(lvr.fee_rate, 1.0, 0.05) + hot * ratio(lvr.fee_rate, 3.0, 0.15)
    assert abs(fee.mean() - want.mean()) < 4 * se(fee - want)
    assert s.fee.mean() == pytest.approx(0.05 + 0.1 * s.spike.mean(), rel=0.01)


def test_ledger_holds_under_moving_fees():
    m = vn.Market(P0=100.0, sigma=0.5, lam=50.0, noise=20.0, size=0.5, hours=4.0)
    q = vn.Quoter(half=0.003, gamma=0.01, limit=5, widen=True)
    rule = fees.Power(0.001, 0.004, 1)
    s = vn.run(m, 0.4, 1e4, rule, q, regimes=fees.Regimes(2.0, 1.0, 1.0), paths=300, seed=6)
    for book in (s.lp, s.mm):
        assert book.inv + book.hedged == pytest.approx(book.total, abs=1e-8 * 1e4)
    edge = s.lp.noise + s.lp.arb
    assert abs(edge.mean() + s.lvr.mean()) < 4 * se(edge + s.lvr)
    assert s.fee.min() > 0.001 + 0.004 * 0.45


def seen_sigma(source, g):
    seen = []

    def rule(sigma):
        seen.append(np.mean(sigma**2))
        return np.full(np.shape(sigma), g)

    vn.run(FAST, math.inf, 1.0, rule, source=source, half_life=1.0, paths=400, seed=3)
    return math.sqrt(np.mean(seen[len(seen) // 2 :]))


def test_market_price_sees_sigma():
    assert seen_sigma("market", 0.05) == pytest.approx(FAST.sigma, rel=0.01)


@pytest.mark.parametrize("g", [0.02, 0.05, 0.2])
def test_pool_price_sees_sigma_squared_times_p_trade(g):
    want = FAST.sigma * math.sqrt(lvr.p_trade(FAST.sigma, FAST.lam, g))
    assert seen_sigma("pool", g) == pytest.approx(want, rel=0.02)
