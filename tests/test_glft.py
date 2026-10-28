import dataclasses
import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from mmsim.glft import Model

PAPER = Model(sigma=0.3, A=0.9, k=0.3, gamma=0.01, Q=30)
BOOK = Model(sigma=1.0, A=2.1, k=3.6, gamma=1.0, Q=10)


def hjb_quotes(m: Model, tau: float):
    q, g, k = m.q, m.gamma, m.k
    c = m.A / (k + g) * (1 + g / k) ** (-k / g)
    cb, ca = c * math.exp(m.skew / 2), c * math.exp(-m.skew / 2)

    def rhs(_, th):
        db = np.append(th[1:] - th[:-1] - (q[:-1] + 1) * m.xi, -np.inf)
        da = np.insert(th[:-1] - th[1:] + (q[1:] - 1) * m.xi, 0, -np.inf)
        return m.mu * q - g * m.sigma**2 * q**2 / 2 + cb * np.exp(k * db) + ca * np.exp(k * da)

    th = solve_ivp(rhs, (0, tau), np.zeros(len(q)), method="Radau", rtol=1e-10, atol=1e-10).y[:, -1]
    base = math.log1p(g / k) / g
    bid = base - (th[1:] - th[:-1] - (q[:-1] + 1) * m.xi)
    ask = base - (th[:-1] - th[1:] + (q[1:] - 1) * m.xi)
    return bid, ask


def test_matches_expm_on_the_paper_example():
    for tau in (0.5, 60.0, 600.0):
        v = PAPER.v(tau)
        w = expm(-tau * PAPER.matrix()) @ np.ones(61)
        assert np.allclose(v, w / w.max(), rtol=1e-11, atol=0)


@pytest.mark.parametrize(
    "m",
    [
        PAPER,
        Model(0.3, 0.9, 0.3, 0.01, 8, mu=0.002),
        Model(0.3, 0.9, 0.3, 0.01, 8, xi=0.4),
        Model(1.0, 2.1, 3.6, 1.0, 10, mu=0.1, xi=0.2),
        Model(0.3, 0.9, 0.3, 0.01, 8, skew=-1.5),
        Model(1.0, 2.1, 3.6, 1.0, 10, mu=0.1, xi=0.2, skew=0.8),
    ],
)
def test_quotes_solve_the_nonlinear_hjb(m):
    tau = 120.0
    bid, ask = m.quotes(tau)
    hb, ha = hjb_quotes(m, tau)
    assert np.allclose(bid[:-1], hb, atol=1e-6)
    assert np.allclose(ask[1:], ha, atol=1e-6)


def test_huge_dynamic_range_stays_positive_and_exact():
    v = BOOK.v(390.0)
    assert v.min() > 0 and v.min() < 1e-15
    bid, ask = BOOK.quotes(390.0)
    hb, ha = hjb_quotes(BOOK, 390.0)
    assert np.allclose(bid[:-1], hb, atol=1e-6)
    assert np.allclose(ask[1:], ha, atol=1e-6)


def test_limits_drop_a_side():
    bid, ask = PAPER.quotes(100.0)
    assert np.isnan(bid[-1]) and np.isnan(ask[0])
    assert not np.isnan(bid[:-1]).any() and not np.isnan(ask[1:]).any()


def test_terminal_quotes_are_the_as_half_spread():
    bid, ask = PAPER.quotes(0.0)
    assert np.allclose(bid[:-1], math.log1p(0.01 / 0.3) / 0.01)
    assert np.allclose(ask[1:], bid[:-1])


def test_near_the_end_it_is_avellaneda_stoikov():
    m, h = PAPER, 1e-4
    bid, ask = m.quotes(h)
    slope = m.gamma * m.sigma**2 / 2
    q = m.q[1:-2]
    assert np.allclose((bid[1:-2] - m.base) / h, (2 * q + 1) * slope, rtol=2e-3)
    q = m.q[2:-1]
    assert np.allclose((ask[2:-1] - m.base) / h, (1 - 2 * q) * slope, rtol=2e-3)


def test_symmetric_without_drift():
    bid, ask = PAPER.quotes(300.0)
    assert np.allclose(bid[:-1], ask[1:][::-1])


def test_quotes_settle_far_from_the_end():
    bid, ask = PAPER.quotes(math.inf)
    b600, a600 = PAPER.quotes(600.0)
    assert np.nanmax(np.abs(bid - b600)) < 1e-6
    assert np.nanmax(np.abs(ask - a600)) < 1e-6


def test_stationary_quotes_come_from_the_ground_state():
    m = Model(0.3, 0.9, 0.3, 0.01, 12)
    lam, U = np.linalg.eigh(m.matrix())
    f0 = np.abs(U[:, 0])
    bid, ask = m.quotes(math.inf)
    assert np.allclose(bid[:-1], m.base + np.log(f0[:-1] / f0[1:]) / m.k, atol=1e-9)
    assert np.allclose(ask[1:], m.base + np.log(f0[1:] / f0[:-1]) / m.k, atol=1e-9)


def test_figures_one_to_three():
    m = Model(0.3, 0.9, 0.3, 0.01, 50)
    window = slice(20, 81)
    for tau in np.linspace(0, 600, 13):
        bid, ask = (x[window] for x in m.quotes(tau))
        assert 1.0 < bid.min() and bid.max() < 5.5
        assert 6.55 < (bid + ask).min() and (bid + ask).max() < 6.64


def test_closed_form_is_close_for_small_inventory():
    bid, ask = PAPER.quotes(math.inf)
    ab, aa = PAPER.approx()
    small = np.abs(PAPER.q) <= 10
    assert np.abs(bid - ab)[small].max() < 0.002
    assert np.abs(ask - aa)[small].max() < 0.002
    assert np.abs(bid - ab)[-2] > 0.5


def test_closed_form_spread():
    m = PAPER
    g, k = m.gamma, m.k
    bid, ask = m.approx()
    psi = 2 / g * math.log1p(g / k) + math.sqrt(
        m.sigma**2 * g / (2 * k * m.A) * (1 + g / k) ** (1 + k / g)
    )
    assert np.allclose(bid + ask, psi)
    assert psi == pytest.approx(6.626, abs=5e-4)


def test_drift_and_impact_closed_forms():
    m = Model(0.3, 0.9, 0.3, 0.01, 30, mu=0.0009)
    bid, ask = m.quotes(math.inf)
    ab, aa = m.approx()
    assert np.abs(bid - ab)[25:36].max() < 0.01
    assert ab[30] < PAPER.approx()[0][30]

    m = Model(0.3, 0.9, 0.3, 0.01, 30, xi=0.5)
    bid, ask = m.quotes(math.inf)
    ab, aa = m.approx()
    assert np.abs(bid - ab)[25:36].max() < 0.01
    assert (bid + ask)[30] == pytest.approx((PAPER.quotes(math.inf)[0] * 2)[30] + 0.5, abs=0.01)


def test_drift_leans_toward_buying():
    m = Model(0.3, 0.9, 0.3, 0.01, 10, mu=0.001)
    bid, ask = m.quotes(math.inf)
    b0, a0 = Model(0.3, 0.9, 0.3, 0.01, 10).quotes(math.inf)
    assert (bid[:-1] < b0[:-1]).all() and (ask[1:] > a0[1:]).all()


def certainty_equivalent(m: Model, T: float, shift: float, n: int, seed: int):
    rng = np.random.default_rng(seed)
    lam, U = np.linalg.eigh(m.matrix())
    w = U.T @ np.ones(len(lam))

    def v(tau, q):
        return (U[q + m.Q] * w * np.exp(-np.outer(tau, lam))).sum(1)

    floor = min(np.nanmin(m.quotes(t)[0]) for t in np.linspace(0, T, 101)) + shift - 0.5
    rmax = m.A * np.exp(-m.k * floor)
    t, q = np.zeros(n), np.zeros(n, int)
    cap, qq = np.zeros(n), np.zeros(n)
    live = np.arange(n)
    while len(live):
        tn = np.minimum(t[live] + rng.exponential(1 / (2 * rmax), len(live)), T)
        qq[live] += q[live] ** 2 * (tn - t[live])
        t[live] = tn
        live = live[tn < T]
        ql, tau = q[live], T - t[live]
        buy = rng.random(len(live)) < 0.5
        nxt = np.clip(np.where(buy, ql + 1, ql - 1), -m.Q, m.Q)
        delta = m.base + np.log(v(tau, ql) / v(tau, nxt)) / m.k + shift
        rate = m.A * np.exp(-m.k * delta)
        assert (rate < rmax).all()
        hit = (rng.random(len(live)) * rmax < rate) & (nxt != ql)
        cap[live[hit]] += delta[hit]
        q[live[hit]] = nxt[hit]
    e = np.exp(-m.gamma * cap + (m.gamma * m.sigma) ** 2 * qq / 2)
    return -np.log(e.mean()) / m.gamma, e.std() / e.mean() / np.sqrt(n) / m.gamma


def test_optimal_quotes_earn_the_value_function():
    m, T = Model(sigma=1.0, A=1.0, k=0.5, gamma=0.05, Q=4), 50.0
    v0 = expm(-T * m.matrix()) @ np.ones(9)
    theory = math.log(v0[m.Q]) / m.k
    ce, se = certainty_equivalent(m, T, 0.0, 20_000, seed=1)
    assert abs(ce - theory) < 3 * se
    for shift in (-0.3, 0.3):
        worse, _ = certainty_equivalent(m, T, shift, 20_000, seed=1)
        assert worse < ce - 5 * se


def test_skew_moves_the_long_run_quotes_by_half_its_log_over_k():
    m = Model(1.0, 2.1, 3.6, 0.3, 5, mu=0.05, xi=0.1)
    b0, a0 = m.quotes(math.inf)
    c0, d0 = m.approx()
    for skew in (-2.0, 0.7):
        tilted = dataclasses.replace(m, skew=skew)
        b1, a1 = tilted.quotes(math.inf)
        assert np.allclose(b1[:-1] - b0[:-1], skew / (2 * m.k), rtol=0, atol=1e-9)
        assert np.allclose(a1[1:] - a0[1:], -skew / (2 * m.k), rtol=0, atol=1e-9)
        c1, d1 = tilted.approx()
        assert np.allclose(c1 - c0, skew / (2 * m.k)) and np.allclose(d1 - d0, -skew / (2 * m.k))
    assert tilted.settle == m.settle
