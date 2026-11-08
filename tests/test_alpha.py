import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.signal import lfilter
from scipy.stats import skellam

from mmsim.alpha import (
    Model,
    always,
    decay,
    fill_fit,
    implied_eta,
    responses,
    simulate,
    solve,
)

BASE = Model(lam=2.0, zeta=0.5, eta=0.2, eps=0.3, width=3.0, phi=0.05, Q=4, n=60)


def closed_form(m, T, a0, q0):
    e = math.exp(-m.zeta * T)
    k = np.arange(-80, 81)
    end = np.abs(q0 + k) @ skellam.pmf(k, m.lam * T, m.lam * T)
    drift = a0 * q0 * (1 - e) / m.zeta
    adverse = 2 * m.lam * m.eps / m.zeta * (T - (1 - e) / m.zeta)
    return m.lam * m.spread * T + drift - adverse - m.spread / 2 * end


@pytest.mark.parametrize("a0, q0", [(0.0, 0), (0.5, 2), (-0.3, 5)])
def test_always_posting_matches_closed_form(a0, q0):
    m = Model(lam=2.0, zeta=0.5, eta=0.2, eps=0.1, width=2.0, Q=40, n=40)
    sol = solve(m, 5.0, always(m))
    assert sol.value(a0, q0) == pytest.approx(closed_form(m, 5.0, a0, q0), abs=0.03)


def test_policy_value_matches_monte_carlo():
    m = replace(BASE, n=120)
    opt = solve(m, 5.0)
    pde = solve(m, 5.0, (opt.bid, opt.ask)).value(0.0, 0)
    mc = simulate(m, opt, 5.0, n_paths=80_000)
    assert abs(mc.mean() - pde) < 3 * mc.std() / math.sqrt(len(mc)) + 0.02


def test_monte_carlo_gap_shrinks_with_the_grid():
    gaps = []
    for n in (30, 120):
        m = replace(BASE, n=n)
        opt = solve(m, 5.0)
        pde = solve(m, 5.0, (opt.bid, opt.ask)).value(0.0, 0)
        gaps.append(pde - simulate(m, opt, 5.0, n_paths=80_000, seed=1).mean())
    assert abs(gaps[1]) < abs(gaps[0]) / 2


def test_impact_is_paid_at_each_fill():
    m = replace(BASE, impact=0.2, eps=0.0, eta=0.0, phi=0.0, Q=30)
    sol = solve(m, 2.0, always(m))
    mc = simulate(m, sol, 2.0, n_paths=40_000)
    assert mc.mean() == pytest.approx(sol.value(0.0, 0), abs=3 * mc.std() / 200)
    flat = solve(replace(m, impact=0.0), 2.0, always(m)).value(0.0, 0)
    assert flat - sol.value(0.0, 0) == pytest.approx(2 * m.lam * 2.0 * 0.2, rel=0.02)


def test_optimum_dominates_fixed_policies():
    opt = solve(BASE, 5.0)
    for policy in (always(BASE), (opt.bid, opt.ask)):
        assert np.all(opt.h >= solve(BASE, 5.0, policy).h - 1e-9)


def test_mirror_symmetry():
    sol = solve(BASE, 5.0)
    assert np.allclose(sol.h, sol.h[::-1, ::-1])
    assert np.array_equal(sol.bid, sol.ask[::-1, ::-1])


def test_ask_is_pulled_as_alpha_rises():
    sol = solve(BASE, 5.0)
    assert np.all(np.diff(sol.ask.astype(int), axis=1) <= 0)
    assert np.all(np.diff(sol.bid.astype(int), axis=1) >= 0)
    assert sol.ask[BASE.Q, 0] and not sol.ask[BASE.Q, -1]


def test_long_inventory_pulls_the_bid_first():
    sol = solve(BASE, 5.0)
    posted = sol.bid.sum(axis=1)
    assert np.all(np.diff(posted) <= 0)
    assert not sol.bid[-1].any()
    assert not sol.ask[0].any()


@pytest.mark.parametrize("impact, posts", [(0.3, True), (0.7, False)])
def test_flat_book_posts_iff_half_spread_beats_impact(impact, posts):
    m = replace(BASE, eps=0.0, eta=0.0, phi=0.0, impact=impact)
    sol = solve(m, 5.0)
    assert sol.post(0.0, 0) == (posts, posts)


def test_policy_settles_with_the_horizon():
    a, b = solve(BASE, 20.0), solve(BASE, 40.0)
    assert np.array_equal(a.bid, b.bid)
    assert np.array_equal(a.ask, b.ask)


def test_post_clips_alpha_and_inventory():
    sol = solve(BASE, 5.0)
    assert sol.post(1e9, 0) == sol.post(BASE.width, 0)
    assert sol.post(0.0, 99) == (False, bool(sol.ask[-1, BASE.n]))


def test_decay_recovers_exact_curve():
    h = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20]) / 60
    b = 1.3 * (1 - np.exp(-25.0 * h))
    c, zeta = decay(h, b)
    assert c == pytest.approx(1.3, rel=1e-4)
    assert zeta == pytest.approx(25.0, rel=1e-4)


def test_responses_of_an_ou_drift():
    rng = np.random.default_rng(3)
    zeta, dt, n = 30.0, 1 / 6000, 400_000
    e = math.exp(-zeta * dt)
    shocks = rng.standard_normal(n) * math.sqrt((1 - e**2) / (2 * zeta))
    a = lfilter([1.0], [1.0, -e], shocks)
    t = np.arange(n) * dt
    mid = np.concatenate([[0.0], np.cumsum(a[:-1] * dt)])
    h = np.array([1, 2, 5, 10, 20, 50]) * dt * 10
    k = slice(0, n - 600, 50)
    b = responses(t[k], a[k] / zeta, lambda x: np.interp(x, t, mid), h)
    c, z = decay(h, b)
    assert c == pytest.approx(1.0, rel=0.05)
    assert z == pytest.approx(zeta, rel=0.1)


def test_implied_eta_from_stationary_variance():
    rng = np.random.default_rng(4)
    zeta, eta, eps, lam, dt, n = 5.0, 1.5, 0.4, 3.0, 1e-3, 1_000_000
    e = math.exp(-zeta * dt)
    sd = eta * math.sqrt((1 - e**2) / (2 * zeta))
    kick = eps * ((rng.random(n) < lam * dt).astype(float) - (rng.random(n) < lam * dt))
    x = rng.standard_normal(n) * sd + kick
    a = lfilter([1.0], [1.0, -e], x)
    assert implied_eta(a[n // 10 :].var(), zeta, eps, lam) == pytest.approx(eta, rel=0.05)


def test_implied_eta_floors_at_zero():
    assert implied_eta(0.01, 1.0, 1.0, 1.0) == 0.0


def test_fill_fit_separates_impact_and_jump():
    rng = np.random.default_rng(5)
    zeta, delta, impact, eps, n = 40.0, 1 / 60, 0.6, -2.0, 50_000
    before = rng.normal(0, 5, n)
    after = before * math.exp(-zeta * delta) + eps + rng.normal(0, 3, n)
    y = impact + before * (1 - math.exp(-zeta * delta)) / zeta + rng.normal(0, 0.5, n)
    w = rng.integers(1, 4, n).astype(float)
    b, e = fill_fit(y, before, after, w, delta, zeta)
    assert b == pytest.approx(impact, abs=0.01)
    assert e == pytest.approx(eps, abs=0.05)
