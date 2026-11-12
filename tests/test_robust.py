import numpy as np
import pytest

from mmsim import avellaneda as av
from mmsim import robust as r

SMALL = r.Game(T=0.2, Q=12)


def test_game_rejects_fill_probability_above_one():
    with pytest.raises(ValueError):
        r.Game(A=(105.0, 250.0))
    with pytest.raises(ValueError):
        r.Game(k=(0.0, 1.5))


def test_symmetric_value_is_closed_form_when_limits_cannot_bind():
    g = r.Game(Q=200)
    V = r.evaluate(g, *r.symmetric(g))
    assert V[0, g.Q] == pytest.approx(g.N * 2 * 140 * g.dt * np.exp(-1) / 1.5, rel=1e-12)


def test_risk_neutral_best_response_quotes_one_over_k():
    g = r.Game(T=0.25, Q=60)
    s = r.best_response(g)
    assert np.allclose(s.bid[:, g.Q], 1 / 1.5, atol=1e-10)
    assert np.allclose(s.ask[:, g.Q], 1 / 1.5, atol=1e-10)


def test_single_stage_equilibrium_closed_form():
    g = r.Game(T=0.005, Q=5)
    s = r.equilibrium(g)
    A, k = g.A[0], g.k[1]
    b = np.where(g.q > 0, g.b[0], np.where(g.q < 0, g.b[1], 0.0))
    assert np.allclose(s.market.b[0], b, atol=1e-9)
    assert np.all(s.market.A == A) and np.all(s.market.k == k)
    inner = slice(1, -1)
    assert np.allclose(s.bid[0, inner], (1 / k - b * g.dt)[inner])
    assert np.allclose(s.ask[0, inner], (1 / k + b * g.dt)[inner])
    c = A * g.dt / (np.e * k)
    want = c * (np.exp(k * b * g.dt) + np.exp(-k * b * g.dt)) + b * g.dt * g.q
    assert np.allclose(s.V[0, inner], want[inner])


def _grid_best(st, i, b, A, k):
    d = np.linspace(0, 3, 601)
    one = st.at(np.array([i]))
    bb, aa = np.meshgrid(d, d, indexing="ij")
    return r.payoff(one, bb.ravel(), aa.ravel(), b, A, k).max()


def test_respond_beats_a_dense_grid():
    g = r.Game(T=0.1, Q=10, zeta=0.02, eta=0.5)
    s = r.best_response(g)
    rng = np.random.default_rng(3)
    for n in rng.integers(0, g.N, 6):
        st = r.stage(g, s.V[n + 1])
        for i in rng.integers(1, len(g.q) - 1, 4):
            b, A, k = rng.uniform(-5, 5), rng.uniform(105, 175), rng.uniform(1.125, 1.875)
            bid, ask = r.respond(st, b, A, k)
            got = r.payoff(st, bid, ask, b, A, k)[i]
            assert got >= _grid_best(st, i, b, A, k) - 1e-12


def test_equilibrium_drift_is_bang_bang_away_from_flat():
    s = r.equilibrium(SMALL, "b")
    q = SMALL.q
    assert np.all(s.market.b[:, q > 0] == SMALL.b[0])
    assert np.all(s.market.b[:, q < 0] == SMALL.b[1])
    assert np.allclose(s.market.b[:, q == 0], 0, atol=1e-9)


@pytest.mark.parametrize("controls", ["b", "bAk"])
def test_equilibrium_is_a_saddle(controls):
    g = r.Game(T=0.2, Q=12, zeta=0.005)
    s = r.equilibrium(g, controls)
    attack = r.worst_case(g, s.bid, s.ask, controls)
    reply = r.best_response(g, s.market)
    assert np.allclose(attack.V, s.V, atol=1e-10)
    assert np.allclose(reply.V, s.V, atol=1e-10)


def test_equilibrium_value_matches_brute_force_minimax_at_a_state():
    g = r.Game(T=0.05, Q=6, zeta=0.01)
    s = r.equilibrium(g)
    st = r.stage(g, s.V[1])
    d = np.linspace(0, 2, 201)
    bb, aa = np.meshgrid(d, d, indexing="ij")
    opts = r.choices(g, "bAk", grid=5)
    for i in (g.Q - 2, g.Q, g.Q + 3):
        one = st.at(np.array([i]))
        g_all = r.payoff(one, bb.ravel()[:, None], aa.ravel()[:, None], *opts)
        maximin = g_all.min(axis=1).max()
        assert maximin == pytest.approx(s.V[0, i], abs=2e-5)


def test_worst_case_is_below_the_fixed_market():
    g = SMALL
    for bid, ask in (r.symmetric(g), r.avellaneda(g, 0.1)):
        fixed = r.evaluate(g, bid, ask)
        worst = r.worst_case(g, bid, ask).V
        assert np.all(worst <= fixed + 1e-12)


def test_equilibrium_has_the_best_guarantee():
    g = SMALL
    eq = r.equilibrium(g).V[0, g.Q]
    for bid, ask in (r.symmetric(g), r.avellaneda(g, 0.1), r.avellaneda(g, 1.0)):
        assert r.worst_case(g, bid, ask).V[0, g.Q] < eq
    fixed = r.best_response(g)
    assert r.worst_case(g, fixed.bid, fixed.ask).V[0, g.Q] < eq


def test_monte_carlo_agrees_with_the_exact_value():
    g = r.Game(T=0.5, Q=20, zeta=0.002, eta=0.05)
    s = r.equilibrium(g)
    run = r.simulate(g, s.bid, s.ask, s.market, paths=40_000, seed=5)
    se = run.reward.std() / np.sqrt(len(run.reward))
    assert abs(run.reward.mean() - s.V[0, g.Q]) < 4 * se
    rn = r.Game(T=0.5, Q=20)
    V = r.evaluate(rn, s.bid, s.ask)
    run = r.simulate(rn, s.bid, s.ask, paths=40_000, seed=6)
    se = run.wealth.std() / np.sqrt(len(run.wealth))
    assert abs(run.wealth.mean() - V[0, rn.Q]) < 4 * se
    assert np.allclose(run.wealth, run.reward)


def test_simulate_respects_inventory_limits_and_per_path_markets():
    g = r.Game(T=0.3, Q=3)
    rng = np.random.default_rng(0)
    mkt = r.Market(rng.uniform(-5, 5, 500), rng.uniform(105, 175, 500), 1.5)
    run = r.simulate(g, *r.symmetric(g), mkt, paths=500, seed=1)
    assert np.abs(run.q).max() <= 3
    assert np.all(run.spread[np.isfinite(run.spread)] == pytest.approx(2 / 1.5))


def test_avellaneda_table_matches_closed_form():
    g = SMALL
    bid, ask = r.avellaneda(g, 0.1)
    n, i = 7, g.Q + 2
    lo, hi = av.quotes(0.0, g.q[i], g.T - n * g.dt, 0.1, g.sigma, 1.5)
    assert bid[n, i] == pytest.approx(max(-lo, 0))
    assert ask[n, i] == pytest.approx(max(hi, 0))
