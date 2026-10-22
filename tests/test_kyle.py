import math

import numpy as np
import pytest

from mmsim import kyle


def test_single_auction_closed_form():
    eq = kyle.solve(1, Sigma0=4.0, sigma_u=0.5)
    assert eq.lam[0] == pytest.approx(math.sqrt(4.0) / (2 * 0.5))
    assert eq.beta[0] == pytest.approx(0.5 / math.sqrt(4.0))
    assert eq.Sigma[1] == pytest.approx(2.0)
    assert eq.profit == pytest.approx(math.sqrt(4.0) * 0.5 / 2)


@pytest.mark.parametrize("N", [2, 5, 40])
def test_lambda_is_the_kalman_gain(N):
    eq = kyle.solve(N, Sigma0=2.0, sigma_u=1.3, T=3.0)
    assert eq.Sigma[0] == pytest.approx(2.0)
    s2 = eq.sigma_u**2
    for n in range(N):
        b, h, prev = eq.beta[n], eq.dt[n], eq.Sigma[n]
        gain = b * h * prev / (b**2 * h**2 * prev + s2 * h)
        assert eq.lam[n] == pytest.approx(gain, rel=1e-10)
        assert eq.Sigma[n + 1] == pytest.approx(prev * (1 - gain * b * h), rel=1e-10)


@pytest.mark.parametrize("N", [1, 3, 25])
def test_value_function_is_the_realised_profit(N):
    eq = kyle.solve(N, Sigma0=1.5, sigma_u=0.8)
    assert kyle.insider_profit(eq, eq.beta) == pytest.approx(eq.profit, rel=1e-12)


def test_equilibrium_beta_is_a_best_response():
    eq = kyle.solve(6, Sigma0=1.0, sigma_u=1.0)
    best = kyle.insider_profit(eq, eq.beta)
    for n in range(6):
        for bump in (0.95, 1.05):
            beta = eq.beta.copy()
            beta[n] *= bump
            assert kyle.insider_profit(eq, beta) < best


def test_continuous_limit():
    eq = kyle.solve(4000, Sigma0=4.0, sigma_u=0.5)
    t = eq.t
    mid = slice(400, 3600)
    assert eq.lam[mid] == pytest.approx(math.sqrt(4.0) / 0.5, rel=2e-3)
    assert eq.Sigma == pytest.approx(4.0 * (1 - t), abs=8e-3)
    assert eq.profit == pytest.approx(math.sqrt(4.0) * 0.5, rel=2e-3)


def test_more_auctions_pay_the_insider_more():
    profits = [kyle.solve(N, 1.0, 1.0).profit for N in (1, 2, 4, 16, 256)]
    assert profits == sorted(profits)
    assert 0.5 == pytest.approx(profits[0]) and profits[-1] < 1.0


def test_monte_carlo_agrees_with_the_solution():
    eq = kyle.solve(4, Sigma0=1.0, sigma_u=1.0)
    paths = kyle.simulate(eq, 10.0, 200_000, seed=1)
    n = len(paths.v)
    err = paths.v[:, None] - paths.p
    assert np.var(err, axis=0) == pytest.approx(eq.Sigma, rel=0.02)
    assert abs(paths.profit.mean() - eq.profit) < 4 * paths.profit.std() / math.sqrt(n)
    dp = np.diff(paths.p, axis=1)
    for k in range(3):
        assert abs(np.corrcoef(dp[:, k], dp[:, k + 1])[0, 1]) < 4 / math.sqrt(n)
        assert abs(np.corrcoef(err[:, k + 1], dp[:, k])[0, 1]) < 4 / math.sqrt(n)
