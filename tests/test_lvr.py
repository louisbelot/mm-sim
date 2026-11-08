import math

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
