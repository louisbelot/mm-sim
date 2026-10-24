import math
import random

import numpy as np
import pytest
from scipy import stats

from mmsim.hawkes import Hawkes, Tracker, _score, decayed_at, fit

PAIR = Hawkes.make([0.4, 0.2], [[1.2, 0.3], [0.8, 0.5]], [[3.0, 1.0], [2.0, 4.0]])


def brute_loglik(h, times, marks, T):
    ll = 0.0
    for k, (t, m) in enumerate(zip(times, marks, strict=True)):
        lam = h.mu[m]
        for s, j in zip(times[:k], marks[:k], strict=True):
            lam += h.alpha[m, j] * math.exp(-h.beta[m, j] * (t - s))
        ll += math.log(lam)
    for i in range(h.d):
        ll -= h.mu[i] * T
        for s, j in zip(times, marks, strict=True):
            ll -= h.alpha[i, j] / h.beta[i, j] * (1 - math.exp(-h.beta[i, j] * (T - s)))
    return ll


def test_loglik_matches_the_double_sum():
    times, marks = PAIR.simulate(150.0, random.Random(1))
    assert 100 < len(times) < 1000
    assert PAIR.loglik(times, marks, 150.0) == pytest.approx(
        brute_loglik(PAIR, times, marks, 150.0), rel=1e-11
    )
    lam = PAIR.intensity(times, marks, times)[np.arange(len(times)), marks]
    assert np.log(lam).sum() - PAIR.compensator(times, marks, [150.0]).sum() == pytest.approx(
        brute_loglik(PAIR, times, marks, 150.0), rel=1e-11
    )


def test_blocked_decay_survives_huge_beta_t():
    rng = np.random.default_rng(2)
    t = np.sort(rng.uniform(0, 5000, 400))
    at = np.sort(np.concatenate([rng.uniform(0, 5000, 300), t[::7], [5001.0]]))
    for beta in (0.01, 0.7, 40.0):
        lag = [q - t[t < q] for q in at]
        r, s = decayed_at(t, at, beta)
        assert np.allclose(r, [np.exp(-beta * x).sum() for x in lag], rtol=1e-11, atol=1e-150)
        assert np.allclose(s, [(x * np.exp(-beta * x)).sum() for x in lag], rtol=1e-9, atol=1e-12)


def test_score_is_the_gradient_of_the_loglik():
    times, marks = PAIR.simulate(300.0, random.Random(7))
    own = [times[marks == i] for i in range(2)]
    args = [PAIR.mu, PAIR.alpha, PAIR.beta]
    _, *grads = _score(*args, own, 300.0)
    for k in range(3):
        for idx in np.ndindex(args[k].shape):
            up = [a.copy() for a in args]
            down = [a.copy() for a in args]
            up[k][idx] += 1e-6
            down[k][idx] -= 1e-6
            fd = (_score(*up, own, 300.0)[0] - _score(*down, own, 300.0)[0]) / 2e-6
            assert grads[k][idx] == pytest.approx(fd, rel=1e-5, abs=1e-6)


def test_mean_rate_and_count_variance_match_theory():
    mu, n, beta = 1.0, 0.5, 2.0
    h = Hawkes.make(mu, n * beta, beta)
    T, w = 100_000.0, 5.0
    times, _ = h.simulate(T, random.Random(3))
    rate = mu / (1 - n)
    assert abs(len(times) / T - rate) < 4 * math.sqrt(mu / (1 - n) ** 3 / T)

    counts = np.bincount((times // w).astype(int), minlength=int(T / w))[: int(T / w)]
    kappa = beta * (1 - n)
    c = rate * n * beta * (2 - n) / (2 * (1 - n))
    var = rate * w + 2 * c * (w / kappa - (1 - math.exp(-kappa * w)) / kappa**2)
    assert counts.var() == pytest.approx(var, rel=0.06)
    assert counts.var() / counts.mean() > 2.5


def test_two_dimensional_rates():
    T = 40_000.0
    times, marks = PAIR.simulate(T, random.Random(4))
    rates = np.bincount(marks, minlength=2) / T
    assert np.allclose(rates, PAIR.mean_rate(), rtol=0.05)


def test_time_rescaling_gives_unit_exponentials():
    times, marks = PAIR.simulate(5000.0, random.Random(5))
    for r in PAIR.residuals(times, marks):
        assert stats.kstest(r, "expon").pvalue > 0.01
    flat = Hawkes.make(PAIR.mean_rate(), np.zeros((2, 2)), 1.0)
    for r in flat.residuals(times, marks):
        assert stats.kstest(r, "expon").pvalue < 1e-6


def test_fit_recovers_the_parameters():
    h = Hawkes.symmetric(2.0, 0.55, 0.15, 5.0)
    T = 20_000.0
    times, marks = h.simulate(T, random.Random(6))
    g = fit(times, marks, T, 2)
    assert np.allclose(g.branching, h.branching, atol=0.04)
    assert np.allclose(g.beta, h.beta, rtol=0.2)
    assert np.allclose(g.mu, h.mu, rtol=0.1)
    assert g.loglik(times, marks, T) >= h.loglik(times, marks, T)


def test_symmetric_keeps_the_mean_rate():
    h = Hawkes.symmetric(0.94, 0.6, 0.1, 6.0)
    assert np.allclose(h.mean_rate(), [0.94, 0.94])
    assert h.radius() == pytest.approx(0.7)


def test_explosive_process_has_no_mean_rate():
    with pytest.raises(ValueError):
        Hawkes.make([1.0, 1.0], [[0.8, 0.4], [0.4, 0.8]], 1.0).mean_rate()
    with pytest.raises(ValueError):
        Hawkes.make([1.0], [[-0.1]], 1.0)


def test_univariate_forecast_closed_form():
    h = Hawkes.make([1.0], [[2.0]], 5.0)
    assert h.forecast()[0, 0] == pytest.approx(1 / 3)
    assert h.forecast(0.4)[0, 0] == pytest.approx((1 - math.exp(-3 * 0.4)) / 3)


def test_forecast_of_counts_from_an_empty_start():
    T, n = 1.5, 20_000
    rng = random.Random(7)
    counts = np.array([np.bincount(PAIR.simulate(T, rng)[1], minlength=2) for _ in range(n)])
    stationary = PAIR.mean_rate() * T
    predicted = stationary + PAIR.forecast(T) @ -PAIR.stationary_state().ravel()
    se = counts.std(axis=0) / math.sqrt(n)
    assert np.all(np.abs(counts.mean(axis=0) - predicted) < 4 * se)
    assert np.all(stationary - predicted > 10 * se)


def test_tracker_intensity_matches_the_event_sum():
    times, marks = PAIR.simulate(60.0, random.Random(3))
    tr = Tracker(PAIR)
    for t, m in zip(times, marks, strict=True):
        tr.update(t, m)
    at = times[-1] + 0.37
    rest = (PAIR.stationary_state() * np.exp(-PAIR.beta * (at - times[0]))).sum(axis=1)
    expected = PAIR.intensity(times, marks, np.array([at]))[0] + rest
    assert tr.intensity(at) == pytest.approx(expected, rel=1e-12)


def test_tracker_excess_is_zero_at_rest_and_follows_the_last_side():
    h = Hawkes.symmetric(0.94, 0.6, 0.1, 6.0)
    tr = Tracker(h)
    assert np.allclose(tr.excess(5.0), 0)
    tr.update(5.0, 1)
    e = tr.excess(5.0)
    assert e[1] > e[0] > 0
    assert np.all(tr.excess(6.0) < e)
    with pytest.raises(ValueError):
        tr.update(4.0, 0)
