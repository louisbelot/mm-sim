import numpy as np
import pytest
from scipy.stats import norm

from mmsim import regime as g
from mmsim import robust as r
from mmsim.ppo import Policy

SMALL = g.World(T=0.3, Q=15, phi=0.01, eta=0.05)


def test_world_rejects_bad_parameters():
    with pytest.raises(ValueError):
        g.World(A=250.0)
    with pytest.raises(ValueError):
        g.World(sigma=(0.0, 2.0))
    with pytest.raises(ValueError):
        g.World(rate=(-1.0, 1.0))
    with pytest.raises(ValueError):
        g.World(pi0=1.5)


def test_update_is_bayes_then_one_markov_step():
    w = g.World(b=(-3.0, 4.0), sigma=(1.0, 3.0), rate=(5.0, 1.0))
    rng = np.random.default_rng(0)
    pi, dz = rng.random(50), 0.3 * rng.standard_normal(50)
    l0 = norm.pdf(dz, w.b[0] * w.dt, w.sigma[0] * np.sqrt(w.dt))
    l1 = norm.pdf(dz, w.b[1] * w.dt, w.sigma[1] * np.sqrt(w.dt))
    post = np.stack([(1 - pi) * l0, pi * l1]) / ((1 - pi) * l0 + pi * l1)
    assert np.allclose(g.update(w, pi, dz), (w.P.T @ post)[1], atol=1e-14)


def test_certain_beliefs_stay_certain_without_switching():
    w = g.World(rate=(0.0, 0.0))
    dz = np.linspace(-1, 1, 7)
    assert np.all(g.update(w, 0.0, dz) == 0.0)
    assert np.all(g.update(w, 1.0, dz) == 1.0)


def test_filter_is_calibrated():
    w = g.World(T=0.5, Q=10, rate=(3.0, 1.0), pi0=0.2)
    run = g.simulate(w, g.table(*r.symmetric(w.game((0.0, 0.0), 4.0))), 40_000, 3, keep=True)
    for n in (1, w.N // 2, w.N - 1):
        s, pi = run.s[n], run.pi[n]
        se = 4 / np.sqrt(len(s))
        assert abs(np.mean(s - pi)) < se
        assert abs(np.mean((s - pi) * pi)) < se
        assert abs(pi.mean() - g.priors(w)[n]) < se


def test_informed_without_switching_is_two_best_responses():
    w = g.World(T=0.3, Q=15, phi=0.01, rate=(0.0, 0.0), sigma=(1.0, 3.0))
    full = g.informed(w)
    for j in (0, 1):
        s = r.best_response(w.game((w.b[j], w.b[j]), w.sigma[j] ** 2), r.Market(w.b[j], w.A, w.k))
        assert np.allclose(full.V[:, j], s.V, atol=1e-9)
        assert np.allclose(full.bid[:, j], s.bid, atol=1e-9)


def test_blind_without_switching_is_a_best_response_to_the_mean_drift():
    w = g.World(T=0.3, Q=15, phi=0.01, rate=(0.0, 0.0), sigma=(1.0, 3.0), pi0=0.3)
    b, s2 = 0.7 * w.b[0] + 0.3 * w.b[1], 0.7 * 1.0 + 0.3 * 9.0
    s = r.best_response(w.game((b, b), s2), r.Market(b, w.A, w.k))
    blind = g.blind(w)
    assert np.allclose(blind.V, s.V, atol=1e-9)
    assert np.allclose(blind.bid, s.bid, atol=1e-9)


def test_blind_value_is_its_value_in_the_hidden_world():
    w = g.World(T=0.3, Q=15, phi=0.01, sigma=(1.0, 3.0), rate=(6.0, 2.0), pi0=0.9)
    blind = g.blind(w)
    V = g.evaluate(w, blind.bid, blind.ask)
    pi = g.priors(w)
    assert np.allclose((1 - pi[:, None]) * V[:, 0] + pi[:, None] * V[:, 1], blind.V, atol=1e-9)


def test_evaluating_the_informed_quotes_returns_their_value():
    full = g.informed(SMALL)
    assert np.allclose(g.evaluate(SMALL, full.bid, full.ask), full.V, atol=1e-9)


def test_uninformative_prices_reduce_bayes_to_a_fixed_drift():
    w = g.World(T=0.3, Q=15, phi=0.01, rate=(0.0, 0.0), sigma=(1e4, 1e4))
    plan, pi = g.bayes(w, 11)
    for i in (0, 3, 7, 10):
        b = (1 - pi[i]) * w.b[0] + pi[i] * w.b[1]
        s = r.best_response(w.game((b, b), 1e8), r.Market(b, w.A, w.k))
        assert np.allclose(plan.V[0, i], s.V[0], atol=1e-6)


def test_certain_bayes_matches_informed():
    w = g.World(T=0.3, Q=15, phi=0.01, rate=(0.0, 0.0))
    plan, _ = g.bayes(w, 21)
    full = g.informed(w)
    assert np.allclose(plan.V[:, 0], full.V[:, 0], atol=1e-9)
    assert np.allclose(plan.V[:, -1], full.V[:, 1], atol=1e-9)


def test_bayes_value_is_convex_in_the_belief():
    plan, _ = g.bayes(SMALL, 51)
    for n in (0, SMALL.N // 2, SMALL.N - 1):
        assert np.diff(plan.V[n], 2, axis=0).min() > -1e-9


def test_information_ordering():
    w = SMALL
    plan, _ = g.bayes(w)
    bayes = g.believed(w, plan)
    full = g.start(w, g.informed(w).V)
    others = [g.minimax(w), g.blind(w)]
    tables = [(s.bid, s.ask) for s in others] + [g.avellaneda(w, 2 * w.phi)]
    values = [g.start(w, g.evaluate(w, *t)) for t in tables]
    assert max(values) < bayes < full


def test_minimax_is_the_drift_game_under_the_worst_volatility():
    w = g.World(T=0.3, Q=15, phi=0.02, sigma=(1.0, 3.0))
    game = r.Game(T=w.T, Q=w.Q, zeta=w.phi * 9.0 * w.dt, eta=w.eta)
    assert np.allclose(g.minimax(w).V, r.equilibrium(game, "b").V, atol=1e-9)


def test_qmdp_is_the_informed_quote_at_a_certain_belief():
    w = g.World(T=0.3, Q=15, phi=0.01, rate=(0.0, 0.0))
    full = g.informed(w)
    quote = g.qmdp(w, full)
    i = np.arange(len(w.q))
    for n in (0, w.N // 2, w.N - 1):
        for j in (0, 1):
            bid, ask = quote(n, i, np.full(len(i), float(j)), None)
            assert np.allclose(bid, full.bid[n, j], atol=1e-9)
            assert np.allclose(ask, full.ask[n, j], atol=1e-9)


@pytest.mark.parametrize("who", ["bayes", "oracle", "blind"])
def test_monte_carlo_agrees_with_the_exact_value(who):
    w = SMALL
    if who == "bayes":
        plan, _ = g.bayes(w)
        quoter, value = g.believer(plan), g.believed(w, plan)
    elif who == "oracle":
        full = g.informed(w)
        quoter, value = g.oracle(full), g.start(w, full.V)
    else:
        s = g.blind(w)
        quoter, value = g.table(s.bid, s.ask), g.start(w, g.evaluate(w, s.bid, s.ask))
    run = g.simulate(w, quoter, 40_000, 11)
    assert abs(run.reward.mean() - value) < 4 * run.reward.std() / np.sqrt(40_000)
    assert np.all(np.abs(run.q) <= w.Q)


def test_env_reward_matches_the_exact_value_of_a_constant_quote():
    w = g.World(T=0.05, Q=3, phi=0.05, eta=0.1, sigma=(1.0, 3.0), rate=(20.0, 5.0))
    env = g.Env(w)
    action = 3 * len(env.deltas) + 2
    bid = np.full((w.N, len(w.q)), env.deltas[3])
    ask = np.full((w.N, len(w.q)), env.deltas[2])
    value = g.start(w, g.evaluate(w, bid, ask))
    total = []
    for seed in range(6000):
        env.reset(seed=seed)
        done, rew = False, 0.0
        while not done:
            _, x, done, _, _ = env.step(action)
            rew += x
        total.append(rew)
    assert abs(np.mean(total) - value) < 4 * np.std(total) / np.sqrt(len(total))


def test_learned_quoter_is_the_greedy_policy():
    w = g.World(Q=10)
    policy = Policy(3, 81, (16,), np.random.default_rng(0))
    quote = g.learned(w, policy)
    i = np.arange(1, 2 * w.Q)
    pi = np.linspace(0.05, 0.95, len(i))
    bid, ask = quote(7, i, pi, None)
    for k in range(len(i)):
        env = g.Env(w)
        env.n, env.q, env.pi = 7, int(i[k] - w.Q), pi[k]
        a = policy.greedy(env.obs())
        assert (bid[k], ask[k]) == (env.deltas[a // 9], env.deltas[a % 9])
