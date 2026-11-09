import math

import numpy as np
import pytest

from mmsim import fees


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
