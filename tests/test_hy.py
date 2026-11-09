import math

import numpy as np
import pytest

from mmsim import hy


def paths(rng, rho=0.6, n=20_000, nx=2000, ny=500, lag=0):
    t = np.linspace(0, 1, n + 1)
    dW = rng.standard_normal((n, 2)) / math.sqrt(n)
    dW[:, 1] = rho * dW[:, 0] + math.sqrt(1 - rho**2) * dW[:, 1]
    W = np.vstack([[0, 0], np.cumsum(dW, axis=0)])
    if lag:
        W[:, 1] = np.concatenate([np.zeros(lag), W[:-lag, 0]])
    ix = np.sort(rng.choice(np.arange(1, n + 1), nx - 1, replace=False))
    iy = np.sort(rng.choice(np.arange(1, n + 1), ny - 1, replace=False))
    ix, iy = np.r_[0, ix], np.r_[0, iy]
    return t[ix], W[ix, 0], t[iy], W[iy, 1]


def test_sample_takes_the_previous_tick():
    t, x = np.array([0.0, 1.0, 2.5]), np.array([10.0, 11.0, 12.0])
    np.testing.assert_array_equal(hy.sample(t, x, [0.0, 0.99, 1.0, 3.0]), [10, 10, 11, 12])
    with pytest.raises(ValueError):
        hy.sample(t, x, [-0.1])


def test_synchronous_observations_give_the_realized_covariance():
    rng = np.random.default_rng(0)
    t = np.sort(rng.random(300))
    x, y = rng.standard_normal(300).cumsum(), rng.standard_normal(300).cumsum()
    assert hy.covariance(t, x, t, y) == pytest.approx(np.diff(x) @ np.diff(y), rel=1e-12)
    assert hy.covariance(t, x, t, x) == pytest.approx(np.sum(np.diff(x) ** 2), rel=1e-12)


def test_hayashi_yoshida_is_unbiased_where_previous_tick_shows_epps():
    rng = np.random.default_rng(1)
    est, prev = [], []
    grid = np.linspace(0, 1, 2001)
    for _ in range(200):
        tx, x, ty, y = paths(rng)
        est.append(hy.covariance(tx, x, ty, y))
        prev.append(np.diff(hy.sample(tx, x, grid)) @ np.diff(hy.sample(ty, y, grid)))
    est, prev = np.array(est), np.array(prev)
    se = est.std() / math.sqrt(len(est))
    assert abs(est.mean() - 0.6) < 4 * se
    assert prev.mean() < 0.6 - 10 * prev.std() / math.sqrt(len(prev))


def test_lead_lag_finds_the_planted_delay():
    rng = np.random.default_rng(2)
    tx, x, ty, y = paths(rng, n=40_000, nx=8000, ny=8000, lag=120)
    lags = np.linspace(-0.006, 0.006, 61)
    ell, c = hy.lead_lag(tx, x, ty, y, lags)
    assert ell == pytest.approx(0.003, abs=1e-9)
    assert c.max() > 5 * np.abs(c[np.abs(lags) > 0.005]).max()
    back, _ = hy.lead_lag(ty, y, tx, x, lags)
    assert back == pytest.approx(-0.003, abs=1e-9)


def test_matrix_cuts_the_window_and_is_symmetric():
    rng = np.random.default_rng(4)
    tx, x, ty, y = paths(rng)
    C = hy.matrix([(tx, x), (ty, y)], (0.25, 0.75))
    keep = (tx >= 0.25) & (tx <= 0.75)
    first = hy.sample(tx, x, [0.25])
    assert C[0, 0] == pytest.approx(np.sum(np.diff(np.r_[first, x[keep]]) ** 2), rel=1e-12)
    assert C[0, 1] == C[1, 0]
    assert abs(C[0, 1] - 0.3) < 0.15


def noisy(rng, n=100_000, nx=10_000, ny=4_000, rho=0.6, omega=0.002, delay=200):
    dW = rng.standard_normal((n, 2)) / math.sqrt(n)
    dW[:, 1] = rho * dW[:, 0] + math.sqrt(1 - rho**2) * dW[:, 1]
    W = np.vstack([[0, 0], np.cumsum(dW, axis=0)])
    W[:, 1] = np.concatenate([np.zeros(delay), W[: len(W) - delay, 1]])
    t = np.linspace(0, 1, n + 1)
    ix = np.r_[0, np.sort(rng.choice(np.arange(1, n + 1), nx, replace=False))]
    iy = np.r_[0, np.sort(rng.choice(np.arange(1, n + 1), ny, replace=False))]
    x = W[ix, 0] + omega * rng.standard_normal(len(ix))
    y = W[iy, 1] + omega * rng.standard_normal(len(iy))
    return t[ix], x, t[iy], y


def test_preaverage_is_the_weighted_sum_of_returns():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(50).cumsum()
    k = 6
    g = hy.triangle(k)
    want = [g @ np.diff(x)[i : i + k - 1] for i in range(len(x) - k)]
    np.testing.assert_allclose(hy.preaverage(x, k), want, rtol=1e-12)
    assert hy.triangle(8).sum() == pytest.approx(8 / 4)


def test_preaveraged_matches_the_double_sum():
    rng = np.random.default_rng(4)
    tx, ty = np.sort(rng.random(300)), np.sort(rng.random(120))
    x, y = rng.standard_normal(300).cumsum(), rng.standard_normal(120).cumsum()
    kx, ky = hy.width(299, 1.0), hy.width(119, 1.0)
    bx, by = hy.preaverage(x, kx), hy.preaverage(y, ky)
    total = sum(
        bx[i] * by[j]
        for i in range(len(bx))
        for j in range(len(by))
        if ty[j] < tx[i + kx] and ty[j + ky] > tx[i]
    )
    want = total / (hy.triangle(kx).sum() * hy.triangle(ky).sum())
    assert hy.preaveraged(tx, x, ty, y) == pytest.approx(want, rel=1e-12)


def test_preaveraging_survives_noise_and_delay():
    rng = np.random.default_rng(5)
    out = np.array(
        [
            (
                hy.covariance(tx, x, ty, y),
                hy.preaveraged(tx, x, ty, y),
                np.sum(np.diff(x) ** 2),
                hy.preaveraged_variance(x),
            )
            for tx, x, ty, y in (noisy(rng) for _ in range(30))
        ]
    )
    mean, se = out.mean(axis=0), out.std(axis=0) / math.sqrt(len(out))
    assert mean[0] < 0.1
    assert abs(mean[1] - 0.6 * 0.998) < 4 * se[1]
    assert mean[2] > 1.05
    assert abs(mean[3] - 1.0) < 4 * se[3]
