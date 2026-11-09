import itertools
import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from mmsim.etf import Band, Dealer, Fund, arbitrage, hedge, simulate, stepwise
from mmsim.multi import Exponential, Logistic, Market


def fund(resid=0.05, seed=1, d=5):
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(d, d))
    w = rng.uniform(0.2, 2.0, d)
    return Fund(w, B @ B.T / d + 0.1 * np.eye(d), resid)


F = fund()
FLOW = Exponential(50, 20)


def dealer(**kw):
    base = dict(fund=F, flow=FLOW, gamma=0.3, z=1.0, names=(0, 2), eta=(0.05, 0.2))
    return Dealer(**{**base, **kw})


def test_hedge_is_the_regression_on_simulated_returns():
    rng = np.random.default_rng(3)
    L = np.linalg.cholesky(F.joint)
    r = rng.standard_normal((200_000, F.d + 1)) @ L.T
    r -= r.mean(axis=0)
    idx = [0, 3, 4]
    beta = np.linalg.lstsq(r[:, 1:][:, idx], r[:, 0], rcond=None)[0]
    h, var = hedge(np.cov(r.T), idx)
    np.testing.assert_allclose(h, -beta, rtol=1e-10)
    assert var == pytest.approx(np.var(r[:, 0] + r[:, 1:][:, idx] @ h, ddof=1), rel=1e-10)


def test_full_basket_hedge_leaves_only_the_residual():
    h, var = hedge(F.joint, range(F.d))
    np.testing.assert_allclose(h, -F.w, atol=1e-12)
    assert var == pytest.approx(F.resid, abs=1e-12)
    assert hedge(F.joint, [])[1] == pytest.approx(F.w @ F.Sigma @ F.w + F.resid)


def test_stepwise_shrinks_variance_and_picks_the_best_single_name():
    chosen, var = stepwise(F.joint, F.d)
    assert sorted(chosen) == list(range(F.d))
    assert np.all(np.diff(var) <= 1e-12)
    assert var[-1] == pytest.approx(F.resid)
    best = min(range(F.d), key=lambda j: hedge(F.joint, [j])[1])
    assert chosen[0] == best
    pairs = min(hedge(F.joint, p)[1] for p in itertools.combinations(range(F.d), 2))
    assert var[2] >= pairs - 1e-12


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"kappa_etf": 0.4, "kappa": 0.1},
        {"kappa": (0.1, 0.3, 0.05, 0.2, 0.15), "redeem": True},
        {"names": tuple(range(5)), "eta": 0.02, "kappa": 0.2, "redeem": True},
        {"flow": Logistic(30, 0.7, 30), "kappa_etf": 1.0},
    ],
)
def test_riccati_solves_the_matrix_ode(kw):
    d = dealer(**kw)
    n = d.m + 1

    def rhs(_, a):
        A = a.reshape(n, n)
        return (d.gamma / 2 * d.Gamma - 2 * A @ np.diag(d.D) @ A).ravel()

    for T in (0.3, 2.0):
        sol = solve_ivp(rhs, (0, T), d.terminal.ravel(), method="DOP853", rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(d.riccati(T), sol.y[:, -1].reshape(n, n), atol=1e-10)
    np.testing.assert_allclose(d.riccati(60.0), d.riccati(math.inf), atol=1e-10)


def test_no_hedge_is_the_single_asset_closed_form():
    d = dealer(names=(), eta=())
    m = Market.of([math.sqrt(F.joint[0, 0])], 0.0, FLOW, d.gamma, z=d.z)
    for tau in (0.5, 3.0, math.inf):
        np.testing.assert_allclose(d.riccati(tau), m.riccati(tau)[0], rtol=1e-12)
    q = np.array([[-2.0], [0.0], [3.0]])
    bid, ask, xi = d.controls(1.0, q)
    mb, ma = m.quotes(q, 1.0)
    np.testing.assert_allclose(bid, mb[:, 0], rtol=1e-12)
    np.testing.assert_allclose(ask, ma[:, 0], rtol=1e-12)
    assert xi.shape == (3, 0)


def test_hedge_target_tends_to_min_variance_as_flow_dries_up():
    target = hedge(F.joint, [0, 2])[0]
    gaps = [np.abs(dealer(flow=Exponential(A, 20)).stationary_hedge() - target).max()
            for A in (50, 1, 1e-2, 1e-4, 1e-6)]
    assert np.all(np.diff(gaps) < 0)
    assert gaps[0] > 0.2
    assert gaps[-1] < 0.005


def test_full_basket_without_residual_targets_the_basket_exactly():
    d = dealer(fund=fund(resid=0.0), names=tuple(range(5)), eta=(0.01, 0.5, 0.1, 2.0, 0.05))
    np.testing.assert_allclose(d.stationary_hedge(), -d.fund.w, rtol=1e-5)


def test_redeeming_a_hedged_position_costs_nothing_at_the_close():
    d = dealer(names=tuple(range(5)), eta=0.1, kappa=0.3, redeem=True)
    x = np.concatenate([[4.0], -4.0 * F.w])
    assert x @ d.terminal @ x == pytest.approx(0.0, abs=1e-12)
    u = dealer(names=tuple(range(5)), eta=0.1, kappa=0.3, kappa_etf=0.3)
    assert x @ u.terminal @ x > 1.0


def cost(r) -> float:
    return (r.hedging + r.penalty + r.close)[0]


def _no_fills(d, scale=1.0):
    def controls(tau, x):
        bid, ask, xi = d.controls(tau, x)
        return np.full_like(bid, np.inf), np.full_like(ask, np.inf), scale * xi

    return controls


@pytest.mark.parametrize("kw", [{"kappa_etf": 0.4, "kappa": 0.1}, {"kappa": 0.1, "redeem": True}])
def test_hedging_cost_without_fills_is_the_quadratic_value(kw):
    d = dealer(flow=Exponential(1e-12, 20), **kw)
    x0 = np.array([2.0, -1.0, 0.5])
    r = simulate(d, 1.0, dt=1e-4, paths=1, x0=x0, controls=_no_fills(d))
    assert r.fills[0] == 0 and r.spread[0] == 0
    assert cost(r) == pytest.approx(x0 @ d.riccati(1.0) @ x0, rel=1e-4)
    for scale in (0.8, 1.2):
        worse = simulate(d, 1.0, dt=1e-4, paths=1, x0=x0, controls=_no_fills(d, scale))
        assert cost(worse) > cost(r) + 1e-3


def test_simulated_ledger_adds_up():
    d = dealer(kappa=0.1, redeem=True)
    r = simulate(d, 0.5, dt=1e-3, paths=400, seed=2, fee=0.01)
    assert r.fills.mean() > 10
    assert abs(r.carry.mean()) < 4 * r.carry.std() / math.sqrt(400)
    np.testing.assert_allclose(r.objective, r.spread + r.carry - r.hedging - r.close - r.penalty)
    x = r.x
    np.testing.assert_allclose(
        r.close, np.einsum("pi,ij,pj->p", x, d.terminal, x) + 0.01 * np.abs(x[:, 0])
    )


@pytest.mark.parametrize("c", [0.0, 0.01, 0.05])
def test_band_closed_forms_match_simulation(c):
    band = Band(s=0.2, lam=400.0, c=c, kappa=1e-6)
    r = arbitrage(band, 500.0, seed=3)
    n = len(r.seen)
    p = r.trades / n
    assert p == pytest.approx(band.p_trade, abs=4 * math.sqrt(p * (1 - p) / n))
    assert r.trades / r.T == pytest.approx(band.trades, rel=0.03)
    assert r.units / r.T == pytest.approx(band.units, rel=0.03)
    assert r.profit / r.T == pytest.approx(band.profit, rel=0.05)
    assert np.sqrt(np.mean(r.seen**2)) == pytest.approx(band.sd, rel=0.03)


def test_band_without_costs_trades_at_every_check():
    band = Band(s=0.2, lam=400.0, c=0.0, kappa=1e-6)
    assert band.p_trade == 1.0
    assert band.profit == pytest.approx(0.2**2 / 2e-6)
    assert band.sd == pytest.approx(0.2 / math.sqrt(400.0))
    wide = Band(s=0.2, lam=400.0, c=0.05, kappa=1e-6)
    assert wide.profit < band.profit and wide.sd > band.sd
