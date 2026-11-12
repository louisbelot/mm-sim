import math
import random

import pytest
from scripted import limit, make, market

from mmsim.book import Book, EventType, OrderType, Reason, Side, Tif
from mmsim.crash import Guard, Maker, Rule, Stops, Value
from mmsim.flow import CSTParams
from mmsim.sim import Agent, Simulator

SKY = CSTParams.sky_perfect()


def ladder(rule, bids=range(100, 89, -1), ask=101):
    g = Guard(rule)
    ids = iter(range(1, 10**6))
    g.submit(next(ids), Side.SELL, ask, 1)
    for px in bids:
        g.submit(next(ids), Side.BUY, px, 1)
    return g, ids


def prints(events):
    return [e.price for e in events if e.type == EventType.TRADE]


def test_no_rule_is_the_plain_book():
    rng = random.Random(7)
    a, b = Book(), Guard()
    live = []
    for oid in range(1, 3000):
        u = rng.random()
        if u < 0.6:
            side = rng.choice(list(Side))
            kind = OrderType.MARKET if rng.random() < 0.15 else OrderType.LIMIT
            kw = dict(
                kind=kind,
                tif=rng.choice(list(Tif)) if kind == OrderType.LIMIT else Tif.IOC,
                post_only=kind == OrderType.LIMIT and rng.random() < 0.1,
            )
            args = (oid, side, rng.randint(95, 105), rng.randint(1, 4))
            assert a.submit(*args, **kw) == b.submit(*args, **kw)
            live.append(oid)
        elif live and u < 0.9:
            oid = live.pop(rng.randrange(len(live)))
            assert a.cancel(oid) == b.cancel(oid)
        elif live:
            oid = rng.choice(live)
            args = (oid, rng.randint(95, 105), rng.randint(1, 4))
            assert a.modify(*args) == b.modify(*args)
    for side in Side:
        assert a.depth(side, 50) == b.depth(side, 50)
    assert b.audit() == ""
    assert b.pauses == []


def test_collar_stops_a_sweep_at_the_band():
    g, ids = ladder(Rule(band=3))
    assert g.ref == 100.5
    out = g.submit(next(ids), Side.SELL, 0, 10, kind=OrderType.MARKET)
    assert prints(out) == [100, 99, 98]
    assert out[-1].type == EventType.CANCELED and out[-1].qty == 7
    assert g.best_bid() == 97
    g.submit(next(ids), Side.SELL, 120, 1)
    assert g.ref == 99


def test_collar_reprices_a_limit_through_the_band():
    g, ids = ladder(Rule(band=3))
    out = g.submit(next(ids), Side.SELL, 50, 10)
    assert prints(out) == [100, 99, 98]
    assert g.best_ask() == 98
    assert g.level(Side.SELL, 98).qty == 7


def test_reference_forgets_old_trades():
    g, ids = ladder(Rule(band=50, window=5))
    g.submit(next(ids), Side.SELL, 0, 2, kind=OrderType.MARKET)
    g.clock = lambda: 3.0
    g.submit(next(ids), Side.SELL, 0, 1, kind=OrderType.MARKET)
    g.submit(next(ids), Side.SELL, 200, 1)
    assert g.ref == pytest.approx((100 + 99 + 98) / 3)
    g.clock = lambda: 6.0
    g.submit(next(ids), Side.SELL, 200, 1)
    assert g.ref == 98


def test_limit_state_turns_into_a_pause():
    g, ids = ladder(Rule(band=3, limit=0.25, pause=5.0))
    g.submit(next(ids), Side.SELL, 0, 10, kind=OrderType.MARKET)
    g.clock = lambda: 0.2
    assert not g.halted()
    g.clock = lambda: 0.3
    g.submit(next(ids), Side.SELL, 110, 1)
    assert g.pauses == [(0.25, 5.25)]
    out = g.submit(next(ids), Side.BUY, 0, 1, kind=OrderType.MARKET)
    assert out[0].type == EventType.REJECTED and out[0].reason == Reason.HALTED
    out = g.submit(next(ids), Side.SELL, 90, 1)
    assert out[0].reason == Reason.HALTED
    assert g.submit(next(ids), Side.BUY, 96, 2)[0].type == EventType.ACCEPTED
    g.clock = lambda: 5.3
    assert not g.halted()
    out = g.submit(next(ids), Side.SELL, 0, 3, kind=OrderType.MARKET)
    assert prints(out) == [97, 96, 96]
    assert len(g.pauses) == 1


def test_limit_state_clears_when_bids_come_back():
    g, ids = ladder(Rule(band=3, limit=0.25))
    g.submit(next(ids), Side.SELL, 0, 10, kind=OrderType.MARKET)
    g.clock = lambda: 0.1
    g.submit(next(ids), Side.BUY, 99, 1)
    g.clock = lambda: 1.0
    assert not g.halted()
    assert g.pauses == []


def test_pause_starts_on_time_with_nothing_happening():
    g, ids = ladder(Rule(band=3, limit=0.25, pause=5.0))
    g.submit(next(ids), Side.SELL, 0, 10, kind=OrderType.MARKET)
    g.clock = lambda: 7.0
    assert not g.halted()
    assert g.pauses == [(0.25, 5.25)]
    assert math.isnan(g.ref)


def test_trip_halts_after_the_trade():
    g, ids = ladder(Rule(trip=5, pause=2.0))
    out = g.submit(next(ids), Side.SELL, 0, 8, kind=OrderType.MARKET)
    assert prints(out) == list(range(100, 92, -1))
    assert g.pauses == [(0.0, 2.0)]
    out = g.submit(next(ids), Side.SELL, 0, 1, kind=OrderType.MARKET)
    assert out[0].reason == Reason.HALTED


def test_clamped_order_keeps_its_new_price():
    class Seller(Agent):
        def on_start(self, ctx):
            self.oid = ctx.sell(50, 1)

    agent = Seller()
    sim = make([], agent, book=lambda: Guard(Rule(band=1)))
    sim.book.clock = lambda: sim.t
    sim.book.submit(next(sim._ids), Side.BUY, 98, 1)
    sim.run(1.0)
    assert sim.contexts[0].orders[agent.oid].price == 100


class Static:
    owner = 0

    def __init__(self, d, levels, sell=0):
        self.d, self.levels, self.sell = d, levels, sell

    def seed(self, book, bid, ask, ids):
        book.submit(next(ids), Side.SELL, 101, 1000)
        for i in range(1, self.levels + 1):
            book.submit(next(ids), Side.BUY, 100 - i, self.d)

    def step(self, book, rng, horizon, ids):
        if self.sell and horizon >= 1.0:
            qty, self.sell = self.sell, 0
            oid = next(ids)
            return 1.0, lambda b: b.submit(oid, Side.SELL, 0, qty, kind=OrderType.MARKET)
        return None


def cascade(X, d, s, levels):
    m = math.ceil(X / d)
    while m <= levels:
        nxt = math.ceil((X + s * m) / d)
        if nxt == m:
            return m
        m = nxt
    return levels


@pytest.mark.parametrize(
    "X,d,s", [(30, 5, 2), (7, 4, 1), (12, 3, 0), (40, 6, 5), (9, 2, 2), (20, 3, 4)]
)
def test_stop_cascade_reaches_the_fixed_point(X, d, s):
    levels = 60
    stops = Stops([(k, s) for k in range(1, levels + 1)] if s else [])
    sim = Simulator(Static(d, levels, X), [stops])
    sim.seed(100, 101)
    sim.run(5.0)
    m = cascade(X, d, s, levels)
    lowest = min(f.price for f in sim.contexts[0].account.fills) if s else None
    bid = sim.book.best_bid()
    if m < levels:
        assert bid == (100 - m if (X + s * m) % d else 100 - m - 1)
        if s:
            assert lowest >= 100 - m
    else:
        assert bid is None
    if s < d:
        assert 0 <= m - X / (d - s) < d / (d - s)


def test_maker_dumps_at_its_hard_limit():
    mm = Maker(half=0.5, size=2, soft=4, hard=4, flat=1, cool=1.5)
    sim = make(
        [(1.0, market(Side.SELL, 2)), (2.0, market(Side.SELL, 2))],
        mm,
    )
    sim.run(3.0)
    ctx = sim.contexts[0]
    assert ctx.position == 1
    dumps = [f for f in ctx.account.fills if not f.maker]
    assert [(f.side, f.price, f.qty) for f in dumps] == [(Side.SELL, 99, 3)]
    assert [o.side for o in ctx.orders.values()] == [Side.SELL]
    sim.run(1.0)
    assert sorted(o.side for o in ctx.orders.values()) == [Side.BUY, Side.SELL]


def test_kill_switch_flattens_and_waits():
    mm = Maker(half=0.5, size=2, soft=2, loss=2.0, rest=4.0)
    sim = make(
        [
            (1.0, market(Side.SELL, 2)),
            (2.0, limit(Side.BUY, 95, 3)),
            (3.0, market(Side.SELL, 5)),
        ],
        mm,
    )
    sim.run(5.0)
    ctx = sim.contexts[0]
    assert mm.kills == [3.0]
    assert ctx.position == 0
    assert not ctx.orders
    sim.run(3.0)
    assert ctx.orders


@pytest.mark.parametrize("kappa,cap,qty", [(2.0, 2, 2), (1.0, 5, 1), (0.5, 5, 0)])
def test_value_trader_buys_the_gap_in_capped_clips(kappa, cap, qty):
    v = Value(every=1.0, kappa=kappa, cap=cap, seed=3)
    script = [
        (0.001, market(Side.SELL, 5)),
        (0.002, limit(Side.BUY, 96)),
        (0.003, limit(Side.SELL, 102, 30)),
    ]
    sim = make(script, v)
    sim.run(5.0)
    fills = sim.contexts[0].account.fills
    assert v.value == 100.5
    clips = {}
    for f in fills:
        clips[f.oid] = clips.get(f.oid, 0) + f.qty
    assert list(clips.values()) == [qty] * len(clips)
    assert all(f.side == Side.BUY and f.price == 102 for f in fills)
    assert bool(fills) == (qty > 0)
