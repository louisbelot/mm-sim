import numpy as np
from scripted import limit, make

from mmsim.book import Side
from mmsim.bots import SymmetricQuoter
from mmsim.flow import CSTParams, ZeroIntelligence
from mmsim.informed import Fundamental, InformedTrader
from mmsim.markout import curve, markouts
from mmsim.sim import Simulator


def test_fundamental_is_brownian_whatever_the_grid():
    rng = np.random.default_rng(0)
    ends = {}
    for grid in ("coarse", "fine"):
        out = []
        for seed in range(3000):
            v = Fundamental(0.7, seed=seed)
            v.start(0.0, 100.0)
            n = 2 if grid == "coarse" else 50
            for t in np.sort(rng.uniform(0, 4, n)):
                v.at(t)
            out.append(v.at(4.0) - 100.0)
        ends[grid] = np.array(out)
    for x in ends.values():
        assert abs(x.mean()) < 4 * 1.4 / np.sqrt(len(x))
        assert abs(x.var() / (0.7**2 * 4) - 1) < 4 * np.sqrt(2 / len(x))


def test_same_time_same_value():
    v = Fundamental(1.0, seed=3)
    v.start(0.0, 50.0)
    assert v.at(2.0) == v.at(2.0) == v.at(1.0)
    assert len(v.path) == 2


def test_buys_until_the_ask_reaches_the_value():
    value = Fundamental(0.0)
    value.start(0.0, 104.0)
    script = [(0.01, limit(Side.SELL, 103)), (0.02, limit(Side.SELL, 104))]
    sim = make(script, InformedTrader(value, rate=20.0, seed=1))
    sim.run(10.0)
    ctx = sim.contexts[0]
    assert [(f.price, f.qty) for f in ctx.account.fills] == [(102, 1)] * 5 + [(103, 1)]
    assert sim.book.best_ask() == 104


def test_value_starts_once_the_book_has_two_sides():
    value = Fundamental(0.0)
    sim = make([(1.0, limit(Side.SELL, 104))], InformedTrader(value, rate=20.0, seed=2))
    sim.book.cancel(2)
    sim.run(3.0)
    (t, v), *_ = value.path
    assert 1.0 <= t < 1.5 and v == 101.5
    assert sim.contexts[0].account.fills == []


def session(seed):
    value = Fundamental(1.0, seed=seed)
    informed = InformedTrader(value, 1.0, seed=seed)
    sim = Simulator(
        ZeroIntelligence(CSTParams.sky_perfect()), [SymmetricQuoter(), informed], seed=seed
    )
    sim.seed(10_000, 10_001)
    sim.warm_up(30)
    sim.run(390)
    return sim, value


def test_informed_flow_picks_off_the_quoter():
    noise, toxic = [], []
    for seed in range(6):
        sim, value = session(seed)
        mm, inf = sim.contexts
        path = dict(value.path)
        for f in inf.account.fills:
            assert (path[f.t] - f.price) * (1 if f.side == Side.BUY else -1) > 0
        for out, hit in ((noise, False), (toxic, True)):
            fills = [f for f in mm.account.fills if (f.counterparty == inf.owner) == hit]
            out.append(curve(fills, markouts(fills, sim.tape, [0.0, 20.0], sim.end))[0])
    noise, toxic = np.mean(noise, axis=0), np.mean(toxic, axis=0)
    assert noise[0] > 0.4 and toxic[0] > 0.4
    assert toxic[1] < 0 < noise[1]
