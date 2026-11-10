import numpy as np
import pytest

from mmsim.detect import Filter, one_sided, stationary, toxic

Q = np.array([[-0.05, 0.03, 0.02], [0.2, -0.2, 0.0], [0.1, 0.0, -0.1]])
RATES = np.array([[1.0, 1.0], [2.5, 1.0], [0.8, 1.9]])


def simulate(Q, rates, T, rng):
    s = rng.choice(len(Q), p=stationary(Q))
    t, events, path = 0.0, [], [(0.0, s)]
    while True:
        out = -Q[s, s]
        total = out + rates[s].sum()
        t += rng.exponential(1 / total)
        if t > T:
            return events, path
        u = rng.random() * total
        if u < out:
            p = Q[s].clip(min=0.0)
            s = rng.choice(len(Q), p=p / p.sum())
            path.append((t, s))
        else:
            events.append((t, int(u - out > rates[s, 0])))


def test_stationary():
    pi = stationary(Q)
    assert pi.sum() == pytest.approx(1.0)
    assert pi @ Q == pytest.approx(np.zeros(3), abs=1e-12)


def test_matches_a_discretized_forward_pass():
    rng = np.random.default_rng(1)
    events, _ = simulate(Q, RATES, 30.0, rng)
    f = Filter(Q, RATES, t=0.0)
    dt = 1e-4
    step = np.eye(3) + (Q - np.diag(RATES.sum(axis=1))) * dt
    p = stationary(Q)
    k = 0
    for i in range(1, 300_001):
        p = p @ step
        while k < len(events) and events[k][0] <= i * dt:
            p = p * RATES[:, events[k][1]]
            f.observe(*events[k])
            k += 1
        p /= p.sum()
        if i % 50_000 == 0:
            assert f.at(i * dt) == pytest.approx(p, abs=2e-3)


def test_without_switching_it_is_a_likelihood_ratio():
    mu, r = 0.9, 0.6
    f = Filter(np.zeros((2, 2)), [[mu, mu], [mu + r, mu]], p0=[0.5, 0.5], t=0.0)
    for t, mark in [(0.3, 0), (0.4, 1), (1.1, 0), (2.5, 0)]:
        f.observe(t, mark)
    p = f.at(4.0)
    assert np.log(p[1] / p[0]) == pytest.approx(3 * np.log1p(r / mu) - r * 4.0)


def test_a_print_of_n_is_n_events():
    a, b = Filter(Q, RATES), Filter(Q, RATES)
    a.observe(1.0, 0, 3)
    for _ in range(3):
        b.observe(1.0, 0)
    assert a.p == pytest.approx(b.p)


def test_posterior_is_calibrated():
    rng = np.random.default_rng(7)
    grid = np.arange(5.0, 400.0, 5.0)
    said, was = [], []
    for _ in range(150):
        events, path = simulate(Q, RATES, 400.0, rng)
        f = Filter(Q, RATES, t=0.0)
        times = np.array([t for t, _ in path])
        k = 0
        for g in grid:
            while k < len(events) and events[k][0] <= g:
                f.observe(*events[k])
                k += 1
            said.append(f.at(g)[1])
            was.append(path[np.searchsorted(times, g, side="right") - 1][1] == 1)
    said, was = np.array(said), np.array(was)
    for lo, hi in [(0.0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.0)]:
        m = (said >= lo) & (said < hi)
        se = np.sqrt(was[m].var() / m.sum())
        assert was[m].mean() == pytest.approx(said[m].mean(), abs=6 * se + 0.01)


def test_one_sided_finds_the_buyer_and_lets_go():
    f = one_sided(1.0, 1.0, 0.01, 0.05, t=0.0)
    assert f.p == pytest.approx([5 / 6, 1 / 12, 1 / 12])
    for t in np.arange(0.125, 10.0, 0.25):
        f.observe(t, 0)
    assert f.p[1] > 0.95 and f.p[2] < 1e-3
    for t in np.arange(10.25, 40.0, 1.0):
        f.observe(t, int(t) % 2)
    assert f.p[0] > 0.9


def test_toxic_without_switching_is_a_likelihood_ratio():
    f = toxic(1.0, 0.5, 0.0, 0.0, t=0.0)
    for t in (0.3, 0.9, 1.0, 2.5):
        f.observe(t, 0)
    f.observe(3.0, 0, 2)
    odds = 2.5**6 / 2**6 * np.exp(-0.5 * 4.0)
    p = f.at(4.0)
    assert p[1] / p[0] == pytest.approx(odds)


def test_toxic_survives_a_long_silence():
    f = toxic(1.0, 3.0, 0.02, 0.06, t=0.0)
    for k in range(30):
        f.observe(0.1 * k, 0)
    assert f.p[1] > 0.9
    late = f.at(1e4)
    assert np.isfinite(late).all() and late[1] < 0.25
    assert late == pytest.approx(f.at(1e3))
