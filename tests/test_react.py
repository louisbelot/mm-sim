import dataclasses
import math

import numpy as np
import pytest
from test_excite import hot
from test_qr import feed, flat

from mmsim import qr
from mmsim.book import Side
from mmsim.bots import SymmetricQuoter
from mmsim.itch import NS_PER_MINUTE, Msg
from mmsim.qr import CANCEL, LIMIT, MARKET, Excite, QueueReactive, Reaction
from mmsim.sim import Latency, Simulator

B, S = Side.BUY, Side.SELL


def at(minute, *args, **kw):
    return Msg(round(minute * NS_PER_MINUTE), *args, **kw)


def test_quotes_are_the_touch_events():
    msgs = [
        at(0, "A", 1, B, 10_000, 3),
        at(0, "A", 2, B, 9_900, 2),
        at(0, "A", 3, S, 10_100, 4),
        at(0, "A", 4, S, 10_200, 1),
        at(1, "D", 1),
        at(2, "A", 5, B, 9_800, 1),
        at(3, "A", 6, B, 9_900, 2),
        at(4, "D", 4),
        at(5, "A", 7, S, 10_100, 2),
        at(6, "E", 3, qty=1),
    ]
    f = qr.fit(msgs, 0.0, 10.0, depth=3, bins=4, aes=[1, 1, 1], edges=(1, 2))
    got = [(f.t[r], s, k) for r, s, k in f.quotes]
    assert got == [(1.0, B, CANCEL), (3.0, B, LIMIT), (5.0, S, LIMIT)]
    assert [(f.t[r], s) for r, s in f.hits] == [(6.0, S)]


def tiny_fit():
    msgs = [at(0, "A", 1, B, 10_000, 3), at(0, "A", 2, S, 10_100, 4), at(1, "D", 1)]
    return qr.fit(msgs, 0.0, 10.0, depth=1, bins=2, aes=[1], edges=(2, 4))


def hand_case(seed):
    rng = np.random.default_rng(seed)
    at_ = np.sort(rng.uniform(0, 5, 60))
    at_[10:13] = at_[10]
    live = rng.random((60, 2)) > 0.15
    hits = np.union1d(rng.choice(60, 20, replace=False), [10])
    by = rng.integers(0, 2, hits.size)
    live[hits, by] = True
    q = np.union1d(rng.choice(60, 25, replace=False), [11, 12])
    qs = rng.integers(0, 2, q.size)
    top = np.zeros((60, 2, 3), dtype=np.int64)
    top[..., 2] = rng.integers(1, 3, (60, 2))
    top[~live, 0] = -1
    quotes = np.column_stack((q, qs, np.full(q.size, CANCEL)))
    f = dataclasses.replace(
        tiny_fit(),
        t=at_,
        h=np.zeros(60, dtype=np.int64),
        top=top,
        hits=np.column_stack((hits, by)),
        quotes=quotes,
    )
    return f, live


def brute_loglik(f, live, rate, own, cross, beta, kind):
    src, by = f.t[f.hits[:, 0]], f.hits[:, 1]
    rows, sides = f.hits.T if kind == MARKET else f.quotes[:, :2].T
    beta, own, cross = map(np.array, (beta, own, cross))

    def kernel(s, upto, strict):
        out = np.zeros_like(beta)
        for m in range(src.size):
            if src[m] < upto or (not strict and src[m] <= upto):
                out += (own if by[m] == s else cross) * np.exp(-beta * (upto - src[m]))
        return out

    ll = 0.0
    for r, s in zip(rows, sides, strict=True):
        if not live[r, s]:
            continue
        lam = rate(f.top[r, s, 2]) + np.sum(beta * kernel(s, f.t[r], True))
        ll += math.log(lam)
    for r in range(f.t.size):
        a, b = f.t[r - 1] if r else 0.0, f.t[r]
        for s in Side:
            if not live[r, s]:
                continue
            ll -= rate(f.top[r, s, 2]) * (b - a)
            for m in range(src.size):
                if src[m] <= a:
                    w = own if by[m] == s else cross
                    ll -= np.sum(w * (np.exp(-beta * (a - src[m])) - np.exp(-beta * (b - src[m]))))
    return ll


@pytest.mark.parametrize("seed", range(3))
def test_loglik_matches_brute_force(seed):
    f, live = hand_case(seed)
    react = Reaction((0.2, 0.05), (0.1, 0.3), 0.6, {1: 1.7})
    e = Excite((0.3, 0.1), (0.2, 0.05), (5.0, 0.7), {CANCEL: react})
    p = dataclasses.replace(flat(1, 2, lambda n: 1.0, lambda n: 2.0 + n, lambda n: 3.0), excite=e)
    got = f.loglik(p, MARKET)
    want = brute_loglik(f, live, lambda n: 3.0, e.own, e.cross, e.beta, MARKET)
    assert got == pytest.approx(want, abs=1e-9)
    got = f.loglik(p, CANCEL)
    psi = {1: 1.7, 2: 0.6}
    want = brute_loglik(
        f, live, lambda n: psi[n] * (2.0 + n), react.own, react.cross, e.beta, CANCEL
    )
    assert got == pytest.approx(want, abs=1e-9)


def test_dead_quotes_drop_out():
    f, live = hand_case(0)
    p = flat(1, 2, lambda n: 1.0, lambda n: 0.0 if n == 2 else 2.0, lambda n: 3.0)
    mu, _, rows, side, key = f._path(p, CANCEL)
    assert np.all(f.top[rows, side, 2] == 1) and np.all(live[rows, side])
    assert np.all(key[f.top[..., 2] == 2] == -1)
    assert np.all(mu[key < 0] == 0)


@pytest.fixture(scope="module")
def calm_fit():
    p = hot()
    f = qr.fit(feed(p, 600.0, 1), 0.0, 600.0, depth=2, bins=8, edges=p.edges)
    return f, f.excite(f.params(regimes=False, min_events=1), min_events=1)


def test_reaction_is_zero_when_quotes_ignore_trades(calm_fit):
    f, p = calm_fit
    got = f.react(p, min_events=200).excite.react
    for kind in (CANCEL, LIMIT):
        r = got[kind]
        assert r.own[0] + r.cross[0] < 0.02
        assert all(abs(v - 1) < 0.1 for v in r.cells.values())


def scaled(p, kind, c):
    r = p.excite.react[kind]
    r = Reaction(
        tuple(c * v for v in r.own),
        tuple(c * v for v in r.cross),
        c * r.rest,
        {k: c * v for k, v in r.cells.items()},
    )
    return dataclasses.replace(p, excite=dataclasses.replace(p.excite, react={kind: r}))


def test_reaction_fit_is_a_maximum(calm_fit):
    f, p = calm_fit
    p = f.react(p, kinds=(CANCEL,), min_events=200)
    n = sum(r.size for r in f.residuals(p, CANCEL))
    ll = [f.loglik(scaled(p, CANCEL, c), CANCEL) for c in (0.999, 1.0, 1.001)]
    assert abs(ll[2] - ll[0]) / 0.002 < 1e-4 * n
    assert ll[1] > max(ll[0], ll[2])
    base = dataclasses.replace(p, excite=dataclasses.replace(p.excite, react=None))
    assert ll[1] >= f.loglik(base, CANCEL)


def test_react_needs_market_kernels(calm_fit):
    f, p = calm_fit
    with pytest.raises(ValueError):
        f.react(dataclasses.replace(p, excite=None))


def echo(cancel=(0.3, 0.05), limit=(0.2, 0.1)):
    react = {CANCEL: Reaction((cancel[0],), (cancel[1],), 1.0, {})}
    react[LIMIT] = Reaction((limit[0],), (limit[1],), 1.0, {})
    p = hot()
    return dataclasses.replace(p, excite=dataclasses.replace(p.excite, react=react))


@pytest.fixture(scope="module")
def echo_fit():
    p = echo()
    return qr.fit(feed(p, 600.0, 2), 0.0, 600.0, depth=2, bins=8, edges=p.edges)


def test_kick_reaches_the_touch_quotes():
    flow = QueueReactive(echo())
    flow.t = 1.0
    flow._kick(Side.BUY)
    x = flow._excess(1.0)
    assert x[MARKET] == pytest.approx([0.4 * 30, 0.1 * 30])
    assert x[CANCEL] == pytest.approx([0.3 * 30, 0.05 * 30])
    assert x[LIMIT] == pytest.approx([0.2 * 30, 0.1 * 30])
    assert flow._excess(1.1)[CANCEL] == pytest.approx([9 * math.exp(-3), 1.5 * math.exp(-3)])


def test_zero_reaction_gives_the_same_tape():
    def run(p):
        sim = Simulator(QueueReactive(p, seed=4), seed=4)
        sim.seed(100, 101)
        sim.run(30.0)
        return sim.tape

    a = run(hot())
    b = run(echo(cancel=(0.0, 0.0), limit=(0.0, 0.0)))
    assert a.mid == b.mid and a.t == b.t


def test_fit_recovers_the_reaction(echo_fit):
    f = echo_fit
    p = f.excite(f.params(regimes=False, min_events=1), min_events=1)
    assert p.excite.own[0] == pytest.approx(0.4, abs=0.04)
    got = f.react(p, min_events=200)
    rates = np.array(got.rates)
    truth = {CANCEL: lambda n: 6.0 + n, LIMIT: lambda n: 30.0 / (1 + 0.3 * n)}
    for kind, (own, cross) in ((CANCEL, (0.3, 0.05)), (LIMIT, (0.2, 0.1))):
        r = got.excite.react[kind]
        assert r.own[0] == pytest.approx(own, abs=0.04)
        assert r.cross[0] == pytest.approx(cross, abs=0.03)
        touch = {k % (got.N + 1): v for k, v in r.cells.items() if k < got.N + 1}
        assert len(touch) >= 6
        for n, psi in touch.items():
            assert psi * rates[0, 0, n, kind] == pytest.approx(truth[kind](n), rel=0.08)


@pytest.mark.parametrize("kind", [CANCEL, LIMIT])
def test_true_reaction_gives_unit_residuals(echo_fit, kind):
    hot_r = np.concatenate(echo_fit.residuals(echo(), kind))
    cold = dataclasses.replace(echo(), excite=dataclasses.replace(echo().excite, react=None))
    cold_r = np.concatenate(echo_fit.residuals(cold, kind))
    assert hot_r.min() >= 0
    assert hot_r.mean() == pytest.approx(1, abs=0.03)
    assert hot_r.var() == pytest.approx(1, abs=0.08)
    assert abs(cold_r.mean() - 1) > 0.05 or cold_r.var() > 1.15


def test_reacting_flow_keeps_the_book_sound():
    bot = SymmetricQuoter(0.5, size=1, max_position=20)
    lag = Latency(feed=0.001, entry=0.001, response=0.001)
    sim = Simulator(QueueReactive(echo(), seed=5), [bot], seed=5, latency={bot: lag})
    sim.seed(100, 101)
    sim.run(20.0)
    assert sim.contexts[0].account.fills
    assert sim.book.audit() == "" and sim.contexts[0]._view.audit() == ""
