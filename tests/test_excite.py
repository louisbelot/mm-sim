import dataclasses
import math
import random

import numpy as np
import pytest
from test_qr import feed, flat, flow_params

from mmsim import qr
from mmsim.book import Book, Side
from mmsim.bots import SymmetricQuoter
from mmsim.qr import MARKET, SHARED, Excite, QueueReactive, _decay, _excite_score
from mmsim.sim import Latency, Simulator

TRUTH = (0.4, 0.1, 30.0)


def flat_market():
    return flat(2, 8, lambda n: 30.0 / (1 + 0.3 * n), lambda n: 6.0 + n, lambda n: 4.0)


def hot(own=0.4, cross=0.1, beta=30.0):
    return dataclasses.replace(flat_market(), excite=Excite((own,), (cross,), (beta,)))


@pytest.fixture(scope="module")
def sim_fit():
    p = hot()
    return qr.fit(feed(p, 600.0, 0), 0.0, 600.0, depth=2, bins=8, edges=p.edges)


def brute(psi, th, M, g_ev, mu, t, side, at, live):
    k = th.size // 3
    own, cross, beta = th[:k], th[k : 2 * k], th[2 * k :]
    ll = -psi @ M
    for j in range(t.size):
        lam = psi[g_ev[j]] * mu[j]
        for m in range(t.size):
            if t[m] < t[j]:
                w = own if side[m] == side[j] else cross
                lam += np.sum(w * beta * np.exp(-beta * (t[j] - t[m])))
        ll += math.log(lam)
    prev = 0.0
    for r, b in enumerate(at):
        for s in Side:
            if not live[r, s]:
                continue
            for m in range(t.size):
                if t[m] <= prev:
                    w = own if side[m] == s else cross
                    decay = np.exp(-beta * (prev - t[m])) - np.exp(-beta * (b - t[m]))
                    ll -= np.sum(w * decay)
        prev = b
    return ll


def score_case(seed):
    rng = np.random.default_rng(seed)
    at = np.sort(rng.uniform(0, 5, 60))
    at[10:13] = at[10]
    rows = np.union1d(rng.choice(60, 25, replace=False), [10])
    side = rng.integers(0, 2, rows.size)
    side[rows == 10] = 0
    live = rng.random((60, 2)) > 0.15
    live[rows, side] = True
    live[11, :] = False
    g_ev = rng.integers(0, 3, rows.size)
    mu = rng.uniform(0.5, 2, rows.size)
    M = rng.uniform(1, 5, 3)
    return M, g_ev, mu, at[rows], side, at, live


@pytest.mark.parametrize("seed", range(3))
def test_score_matches_brute_force(seed):
    case = score_case(seed)
    psi = np.array([0.7, 1.1, 0.4])
    th = np.array([0.3, 0.2, 0.1, 0.05, 5.0, 0.7])
    ll, g_psi, g = _excite_score(psi, th, *case)
    assert ll == pytest.approx(brute(psi, th, *case), abs=1e-9)
    eps = 1e-6
    for x, grad in ((psi, g_psi), (th, g)):
        for i in range(x.size):
            up, down = x.copy(), x.copy()
            up[i] += eps
            down[i] -= eps
            args = (up, th) if x is psi else (psi, up)
            back = (down, th) if x is psi else (psi, down)
            num = (brute(*args, *case) - brute(*back, *case)) / (2 * eps)
            assert grad[i] == pytest.approx(num, abs=1e-5)


@pytest.mark.parametrize("beta", [0.01, 30.0, 3e5])
def test_decay_matches_brute_force(beta):
    rng = np.random.default_rng(4)
    t = np.sort(np.concatenate([rng.uniform(0, 50, 300), 20 + rng.uniform(0, 1e-4, 40)]))
    at = np.sort(np.concatenate([rng.uniform(0, 50, 200), t[::5]]))
    r, q = _decay(t, at, beta)
    lag = [a - t[t < a] for a in at]
    assert np.allclose(r, [np.exp(-beta * x).sum() for x in lag], rtol=1e-11, atol=1e-20)
    assert np.allclose(q, [(x * np.exp(-beta * x)).sum() for x in lag], rtol=1e-9, atol=1e-20)


def test_zero_kernel_gives_the_same_tape():
    def run(p):
        sim = Simulator(QueueReactive(p, seed=3), seed=3)
        sim.seed(1000, 1001)
        sim.run(5.0)
        return sim.tape

    p = flow_params()
    a = run(p)
    b = run(dataclasses.replace(p, excite=Excite((0.0,), (0.0,), (30.0,))))
    assert a.mid == b.mid and a.t == b.t


def test_kick_decays_at_beta():
    p = hot(beta=20.0)
    flow = QueueReactive(p)
    flow.t = 1.0
    flow._kick(Side.BUY)
    assert flow._excess(1.0) == pytest.approx([0.4 * 20.0, 0.1 * 20.0])
    assert flow._excess(1.05) == pytest.approx([8.0 * math.exp(-1.0), 2.0 * math.exp(-1.0)])
    flow.t = 1.05
    flow._kick(Side.SELL)
    e = math.exp(-1.0)
    assert flow._excess(1.05) == pytest.approx([20 * (0.4 * e + 0.1), 20 * (0.1 * e + 0.4)])


def test_branching_sets_the_market_rate(sim_fit):
    rate = len(sim_fit.hits) / sim_fit.T
    mu, rho = 2 * 4.0, 0.5
    sd = math.sqrt(mu / (1 - rho) ** 3 / sim_fit.T)
    assert abs(rate - mu / (1 - rho)) < 4 * sd
    calm = qr.fit(feed(flat_market(), 600.0, 0), 0.0, 600.0, depth=2, bins=8)
    assert len(calm.hits) / calm.T == pytest.approx(mu, rel=0.05)


def test_fit_recovers_the_kernel(sim_fit):
    p = sim_fit.excite(sim_fit.params(regimes=False, min_events=1), min_events=1)
    own, cross, beta = TRUTH
    assert p.excite.own[0] == pytest.approx(own, abs=0.04)
    assert p.excite.cross[0] == pytest.approx(cross, abs=0.03)
    assert p.excite.beta[0] == pytest.approx(beta, abs=4.0)
    base = np.array(p.rates)[0, 0, 1:7, MARKET]
    assert np.all(np.abs(base / 4.0 - 1) < 0.1)


def test_excite_works_with_a_clock(sim_fit):
    p = sim_fit.excite(sim_fit.params(regimes=False, min_events=1, clock=SHARED), min_events=1)
    assert p.clock is not None
    assert 0.35 < p.excite.branching < 0.65


def test_excite_refuses_other_edges(sim_fit):
    p = dataclasses.replace(sim_fit.params(regimes=False), edges=(1, 9))
    with pytest.raises(ValueError):
        sim_fit.excite(p)


def test_excited_flow_never_draws_an_unseen_market_size():
    p = hot()
    rates = np.array(p.rates)
    rates[1, ..., MARKET] = 0.0
    sizes = dict(p.sizes)
    sizes[MARKET, 1] = ()
    p = dataclasses.replace(p, rates=tuple(map(tuple, rates.tolist())), sizes=sizes)
    book, rng, ids = Book(), random.Random(1), iter(range(1, 10**9))
    flow = QueueReactive(p, seed=1)
    flow.seed(book, 100, 101, ids)
    for _ in range(3000):
        step = flow.step(book, rng, 1e9, ids)
        step[1](book)
    assert book.audit() == ""


def test_excited_flow_runs_under_latency():
    p = dataclasses.replace(flow_params(), excite=Excite((0.5,), (0.1,), (200.0,)))
    bot = SymmetricQuoter(0.5, size=5, max_position=50)
    lag = Latency(feed=0.001, entry=0.001, response=0.001)
    sim = Simulator(QueueReactive(p, seed=3), [bot], seed=3, latency={bot: lag})
    sim.seed(1000, 1001)
    sim.run(5.0)
    assert sim.contexts[0].account.fills
    assert sim.book.audit() == "" and sim.contexts[0]._view.audit() == ""


def test_a_traded_cell_with_no_rate_gets_the_level_rate(sim_fit):
    p = sim_fit.params(regimes=False, min_events=1)
    rates = np.array(p.rates)
    rates[0, :, 1, MARKET] = 0.0
    p = dataclasses.replace(p, rates=tuple(map(tuple, rates.tolist())))
    got = sim_fit.excite(p, min_events=10**9)
    assert np.all(np.array(got.rates)[0, :, 1, MARKET] > 0)
    assert 0.35 < got.excite.branching < 0.65
