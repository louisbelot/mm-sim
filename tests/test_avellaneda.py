import numpy as np
import pytest
from scipy.optimize import brentq, minimize_scalar

from mmsim import avellaneda as av

PAPER = {
    (0.1, "inventory"): (1.49, 65.0, 6.6, 0.08, 2.9),
    (0.1, "symmetric"): (1.49, 68.4, 12.7, 0.26, 8.4),
    (0.01, "inventory"): (1.35, 68.6, 8.7, 0.12, 5.1),
    (0.01, "symmetric"): (1.35, 68.8, 12.8, 0.09, 8.7),
    (1.0, "inventory"): (3.02, 31.4, 5.0, 0.02, 1.7),
    (1.0, "symmetric"): (3.02, 44.0, 11.0, 0.00, 5.1),
}


@pytest.mark.parametrize("gamma,strategy", PAPER)
def test_reproduces_paper_tables(gamma, strategy):
    spread, profit, sd_profit, q, sd_q = PAPER[gamma, strategy]
    n = 20_000
    run = av.simulate(gamma, strategy, n_paths=n, seed=7)

    def se_mean(sd):
        return sd * np.sqrt(1 / 1000 + 1 / n)

    def se_std(sd):
        return sd * np.sqrt(1 / 2000 + 1 / (2 * n))

    assert run.spread == pytest.approx(spread, abs=0.005)
    assert abs(run.profit.mean() - profit) < 4 * se_mean(sd_profit) + 0.05
    assert abs(run.profit.std() - sd_profit) < 4 * se_std(sd_profit) + 0.05
    assert abs(run.final_q.mean() - q) < 4 * se_mean(sd_q) + 0.005
    assert abs(run.final_q.std() - sd_q) < 4 * se_std(sd_q) + 0.05


def test_reservation_price_is_the_frozen_inventory_indifference_price():
    gamma, sigma, s, tau = 0.1, 2.0, 100.0, 0.7

    def log_minus_v(x, q):
        return -gamma * x - gamma * q * s + (gamma * q * sigma) ** 2 * tau / 2

    for q in (-3, 0, 2, 7):
        rb = brentq(lambda r, q=q: log_minus_v(-r, q + 1) - log_minus_v(0, q), 0, 500)
        ra = brentq(lambda r, q=q: log_minus_v(r, q - 1) - log_minus_v(0, q), 0, 500)
        assert av.reservation_price(s, q, tau, gamma, sigma) == pytest.approx((ra + rb) / 2)


def test_spread_term_maximises_the_gain_from_one_fill():
    gamma, k, A = 0.1, 1.5, 140.0
    best = minimize_scalar(
        lambda d: -A * np.exp(-k * d) * (1 - np.exp(-gamma * d)),
        bounds=(0, 10),
        method="bounded",
        options={"xatol": 1e-10},
    )
    assert av.spread(0.0, gamma, 2.0, k) == pytest.approx(2 * best.x, abs=1e-6)


def test_risk_neutral_limit():
    assert av.spread(0.5, 1e-9, 2.0, 1.5) == pytest.approx(2 / 1.5)
    assert av.reservation_price(100.0, 5, 0.5, 1e-9, 2.0) == pytest.approx(100.0)


def test_quotes_skew_against_inventory():
    bid0, ask0 = av.quotes(100.0, 0, 0.5, 0.1, 2.0, 1.5)
    bid, ask = av.quotes(100.0, 3, 0.5, 0.1, 2.0, 1.5)
    assert (bid0 + ask0) / 2 == pytest.approx(100.0)
    assert ask - bid == pytest.approx(ask0 - bid0)
    assert bid < bid0 and ask < ask0
    assert bid0 - bid == pytest.approx(3 * 0.1 * 4.0 * 0.5)


def test_quotes_work_on_arrays():
    bid, ask = av.quotes(np.array([100.0, 101.0]), np.array([0, -2]), 0.25, 0.1, 2.0, 1.5)
    assert bid.shape == ask.shape == (2,)
    assert ask[1] - ask[0] == pytest.approx(1 + 2 * 0.1 * 4.0 * 0.25)


def test_same_seed_same_market():
    a = av.simulate(0.1, "inventory", n_paths=50, seed=3)
    b = av.simulate(0.1, "symmetric", n_paths=50, seed=3)
    assert np.array_equal(a.mid, b.mid)
    with pytest.raises(ValueError):
        av.simulate(0.1, "momentum")
