import math

import numpy as np
import pytest
from scipy.integrate import quad

from mmsim import lvr


def stationary(fn, sigma, lam, gamma):
    zeta = math.sqrt(2 * lam) / sigma

    def tails(x):
        return (fn(gamma + x, gamma) + fn(-gamma - x, gamma)) / 2 * math.exp(-zeta * x)

    top = 60 / (zeta - 0.5)
    return lam * lvr.p_trade(sigma, lam, gamma) * zeta * quad(tails, 0, top, limit=200)[0]


@pytest.mark.parametrize("sigma,lam,gamma", [(1.0, 50.0, 0.05), (0.8, 3.0, 0.3), (2.0, 1.0, 0.01)])
def test_closed_forms_match_theorem_2_integrals(sigma, lam, gamma):
    assert lvr.arb_rate(sigma, lam, gamma) == pytest.approx(
        stationary(lvr.arb_profit, sigma, lam, gamma), rel=1e-8
    )
    assert lvr.fee_rate(sigma, lam, gamma) == pytest.approx(
        stationary(lvr.fee_income, sigma, lam, gamma), rel=1e-8
    )


def test_no_fee_fast_blocks_recover_lvr():
    assert lvr.arb_rate(1.0, 1e9, 0.0) == pytest.approx(lvr.lvr_rate(1.0), rel=1e-8)
    assert lvr.fee_rate(1.0, 1e9, 0.0) == 0


def test_fees_and_arbitrage_split_lvr_when_blocks_are_fast():
    sigma, gamma = 0.5, 0.002
    for lam in (1e4, 1e6):
        pt = lvr.p_trade(sigma, lam, gamma)
        assert lvr.arb_rate(sigma, lam, gamma) / lvr.lvr_rate(sigma) == pytest.approx(pt, rel=1e-3)
        assert lvr.fee_rate(sigma, lam, gamma) / lvr.lvr_rate(sigma) == pytest.approx(
            1 - pt, rel=1e-3
        )


def test_slow_blocks_blow_up():
    assert lvr.arb_rate(4.0, 2.0, 0.1) == math.inf
    assert lvr.fee_rate(4.0, 2.0, 0.1) == math.inf
    assert lvr.stale_quote_loss(2.0, 1.0, 0.01) == math.inf


@pytest.mark.parametrize("gamma", [0.0, 0.05, 0.2, 0.5])
def test_mispricing_chain_matches_closed_forms(gamma):
    sigma, lam = 1.0, 50.0
    c = lvr.mispricing(sigma, lam, gamma, chains=2000, blocks=1000, seed=4)
    assert abs(c.arb - lvr.arb_rate(sigma, lam, gamma)) < 4 * c.arb_se
    assert abs(c.fee - lvr.fee_rate(sigma, lam, gamma)) < 4 * c.fee_se + 1e-12
    n = 2000 * 1000
    pt = lvr.p_trade(sigma, lam, gamma)
    assert abs(c.trade - pt) < 4 * math.sqrt(pt * (1 - pt) / n) + 1e-3


def test_fixed_blocks_trade_less_than_poisson():
    sigma, lam, gamma = 1.0, 50.0, 0.2
    poisson = lvr.mispricing(sigma, lam, gamma, chains=1000, blocks=500, seed=1)
    fixed = lvr.mispricing(sigma, lam, gamma, chains=1000, blocks=500, fixed=True, seed=1)
    assert fixed.arb < poisson.arb - 4 * poisson.arb_se


def test_stale_quote_loss_is_laplace_tail():
    sigma, lam, h = 0.02, 400.0, 0.003
    b = sigma / math.sqrt(2 * lam)
    rng = np.random.default_rng(2)
    X = sigma * np.sqrt(rng.exponential(1 / lam, 2_000_000)) * rng.standard_normal(2_000_000)
    loss = np.maximum(np.exp(X) - 1 - h, 0) + np.maximum(1 - h - np.exp(X), 0)
    assert lvr.stale_quote_loss(sigma, lam, h) == pytest.approx(loss.mean(), rel=0.02)
    assert lvr.stale_quote_loss(sigma, lam, h) == pytest.approx(b * math.exp(-h / b), rel=0.05)
