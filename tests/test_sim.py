import math

import numpy as np
import pytest
from scripted import limit, make, market

from mmsim.book import Side, Tif
from mmsim.flow import CSTParams, ZeroIntelligence, prob_fill_first
from mmsim.sim import Agent, Simulator


class Recorder(Agent):
    def __init__(self, on_start=None):
        self.start = on_start
        self.fills = []
        self.books = 0

    def on_start(self, ctx):
        if self.start:
            self.start(ctx)

    def on_fill(self, ctx, fill):
        self.fills.append(fill)

    def on_book(self, ctx):
        self.books += 1


def test_resting_order_fills_against_market_orders():
    agent = Recorder(lambda ctx: ctx.buy(100, 2))
    sim = make([(1.0, market(Side.SELL, 1)), (2.0, market(Side.SELL, 3))], agent)
    ctx = sim.contexts[0]

    sim.run(1.5)
    ((oid, order),) = ctx.orders.items()
    assert order.qty == 1 and ctx.position == 1 and ctx.account.cash == -100
    f = agent.fills[0]
    assert (f.t, f.oid, f.side, f.price, f.qty, f.maker) == (1.0, oid, Side.BUY, 100, 1, True)
    assert f.mid == 101

    sim.run(1.0)
    assert ctx.orders == {} and ctx.position == 2
    assert sim.book.best_bid() == 99 and sim.book.find(1).qty == 3
    assert agent.books == 2


def test_taker_fill_and_ioc_remainder():
    agent = Recorder(lambda ctx: ctx.buy(103, 7, tif=Tif.IOC))
    sim = make([], agent, taker_fee=0.5)
    sim.run(1.0)
    ctx = sim.contexts[0]
    assert [(f.price, f.qty, f.maker, f.fee) for f in agent.fills] == [(102, 5, False, 2.5)]
    assert ctx.orders == {} and ctx.position == 5
    assert ctx.account.cash == -5 * 102 - 2.5


def test_position_is_current_inside_the_callback_but_fills_wait():
    seen = []

    class Taker(Agent):
        def on_start(self, ctx):
            ctx.buy(102, 1)
            seen.append(("after buy", ctx.position))

        def on_fill(self, ctx, fill):
            seen.append(("fill", fill.qty))

    sim = make([], Taker())
    sim.run(1.0)
    assert seen == [("after buy", 1), ("fill", 1)]


def test_whole_event_batch_is_booked_before_callbacks():
    seen = []

    class TwoBids(Agent):
        def on_start(self, ctx):
            ctx.buy(100, 1)
            ctx.buy(100, 1)

        def on_fill(self, ctx, fill):
            seen.append((fill.oid, ctx.position, len(ctx.orders)))

    sim = make([(1.0, market(Side.SELL, 2))], TwoBids())
    sim.run(2.0)
    assert seen == [(3, 2, 0), (4, 2, 0)]


def test_quotes_and_mark_ignore_own_orders():
    def start(ctx):
        ctx.buy(100, 1)
        ctx.sell(101, 1)

    sim = make([], Recorder(start))
    sim.run(0.5)
    ctx = sim.contexts[0]
    assert (sim.book.best_bid(), sim.book.best_ask()) == (100, 101)
    assert (ctx.bid, ctx.ask, ctx.mid) == (99, 102, 100.5)


def test_cannot_touch_other_orders():
    sim = make([], Recorder(lambda ctx: ctx.cancel(1)))
    sim.run(1.0)
    assert sim.book.find(1).qty == 5


def test_modify_and_cancel_track_open_orders():
    def start(ctx):
        oid = ctx.buy(100, 4)
        ctx.modify(oid, 101, 2)
        assert ctx.orders[oid].price == 101 and ctx.orders[oid].qty == 2
        ctx.modify(oid, 102, 2)
        assert oid not in ctx.orders and ctx.position == 2
        other = ctx.sell(105, 1)
        ctx.cancel_all()
        assert other not in ctx.orders and other not in ctx.book

    sim = make([], Recorder(start))
    sim.run(1.0)


def test_rejected_order_is_forgotten():
    agent = Recorder(lambda ctx: ctx.buy(100, 0))
    sim = make([], agent)
    sim.run(1.0)
    assert sim.contexts[0].orders == {} and sim._agent_orders == {}


def test_timers_fire_in_order():
    times = []

    class Clock(Agent):
        def on_start(self, ctx):
            ctx.set_timer(0.25)
            ctx.set_timer(0.75)

        def on_timer(self, ctx):
            times.append(ctx.t)

    sim = make([(0.5, limit(Side.BUY, 98))], Clock())
    sim.run(1.0)
    assert times == [0.25, 0.75]
    assert sim.t == 1.0 and sim.book.best_bid() == 99


def test_samples_on_a_fixed_grid():
    agent = Recorder(lambda ctx: ctx.buy(100, 1))
    sim = make([(1.5, market(Side.SELL, 1)), (2.5, limit(Side.SELL, 101, 2))], agent)
    sim.run(4.0)
    h = sim.contexts[0].history
    assert [s.t for s in h] == [0, 1, 2, 3, 4]
    assert [s.position for s in h] == [0, 0, 1, 1, 1]
    assert [s.mid for s in h] == [100.5, 100.5, 100.5, 100.0, 100.0]
    assert h[-1].pnl == sim.contexts[0].pnl() == 0


def test_warm_up_runs_flow_without_agents():
    agent = Recorder()
    sim = make([(0.5, limit(Side.BUY, 100))], agent)
    sim.warm_up(1.0)
    assert agent.books == 0 and sim.book.best_bid() == 100
    sim.run(1.0)
    with pytest.raises(RuntimeError):
        sim.warm_up(1.0)


class QueueRacer(Agent):
    def __init__(self):
        self.filled = None

    def on_start(self, ctx):
        ctx.buy(100, 1)

    def on_fill(self, ctx, fill):
        if self.filled is None:
            self.filled = True
            ctx.stop()

    def on_book(self, ctx):
        if self.filled is None and ctx.ask != 101:
            self.filled = False
            ctx.stop()


@pytest.mark.parametrize("a,b", [(1, 1), (3, 1), (2, 3)])
def test_queue_race_matches_exact(a, b):
    sky = CSTParams.sky_perfect()
    n, fills = 1500, 0
    for rep in range(n):
        sim = Simulator(ZeroIntelligence(sky), [racer := QueueRacer()], seed=rep, record_every=None)
        bg = iter(range(10**6, 10**7))
        for _ in range(b - 1):
            sim.book.submit(next(bg), Side.BUY, 100, 1)
        for _ in range(a):
            sim.book.submit(next(bg), Side.SELL, 101, 1)
        for j in range(1, 10):
            for _ in range(2):
                sim.book.submit(next(bg), Side.BUY, 100 - j, 1)
                sim.book.submit(next(bg), Side.SELL, 101 + j, 1)
        sim.run(math.inf)
        fills += racer.filled
    p = prob_fill_first(a, b, sky.lam[0], sky.mu, sky.theta[0])
    assert abs(fills / n - p) < 4 * math.sqrt(p * (1 - p) / n)


def test_cst_session_keeps_book_and_accounts_consistent():
    class Joiner(Agent):
        def on_book(self, ctx):
            sides = {o.side for o in ctx.orders.values()}
            if Side.BUY not in sides and ctx.bid is not None:
                ctx.buy(ctx.bid, 1, post_only=True)
            if Side.SELL not in sides and ctx.ask is not None:
                ctx.sell(ctx.ask, 1, post_only=True)

    sim = Simulator(ZeroIntelligence(CSTParams.sky_perfect()), [Joiner()], seed=2)
    sim.seed(1000, 1001)
    sim.warm_up(30)
    sim.run(300)
    ctx = sim.contexts[0]
    assert sim.book.audit() == ""
    assert len(ctx.account.fills) > 20
    resting = {o.id: o for side in Side for o in sim.book.orders(side) if o.owner == ctx.owner}
    assert {oid: (o.side, o.price, o.qty) for oid, o in ctx.orders.items()} == {
        oid: (o.side, o.price, o.qty) for oid, o in resting.items()
    }
    buys = sum(f.qty for f in ctx.account.fills if f.side == Side.BUY)
    sells = sum(f.qty for f in ctx.account.fills if f.side == Side.SELL)
    assert ctx.position == buys - sells
    assert len(ctx.history) == 301


def test_fills_know_their_counterparty():
    sim = make([(1.0, market(Side.SELL, 1))], Recorder(lambda ctx: ctx.buy(100, 1)))
    sim.run(2.0)
    assert [f.counterparty for f in sim.contexts[0].account.fills] == [sim.flow.owner]

    buyer = Recorder(lambda ctx: ctx.buy(100, 1))
    seller = Recorder(lambda ctx: ctx.sell(100, 1, tif=Tif.IOC))
    sim = make([], buyer, seller)
    sim.run(1.0)
    a, b = sim.contexts
    assert [(f.maker, f.counterparty) for f in a.account.fills] == [(True, b.owner)]
    assert [(f.maker, f.counterparty) for f in b.account.fills] == [(False, a.owner)]


def test_self_trade_books_both_sides():
    def cross(ctx):
        ctx.buy(100, 1)
        ctx.sell(100, 1)

    sim = make([], Recorder(cross))
    sim.run(1.0)
    ctx = sim.contexts[0]
    fills = ctx.account.fills
    assert [(f.side, f.maker, f.counterparty) for f in fills] == [
        (Side.BUY, True, ctx.owner),
        (Side.SELL, False, ctx.owner),
    ]
    assert ctx.position == 0 and ctx.orders == {}


def test_tape_follows_the_book_mid():
    seen = []

    class Watcher(Agent):
        def on_book(self, ctx):
            seen.append((ctx.t, ctx.book.mid()))

    sim = Simulator(ZeroIntelligence(CSTParams.sky_perfect()), [Watcher()], seed=5)
    sim.seed(1000, 1001)
    sim.run(60)
    t, mid = zip(*[(t, m) for t, m in seen if m is not None], strict=True)
    assert list(sim.tape.at(t)) == list(mid)
    assert len(sim.tape.t) < len(seen)
    assert np.all(np.diff(sim.tape.t) > 0)


def test_fills_remember_their_place_in_the_queue():
    def start(ctx):
        ctx.buy(99, 1)
        ctx.buy(99, 2)

    agent = Recorder(start)
    sim = make([(1.0, market(Side.SELL, 3)), (2.0, market(Side.SELL, 4))], agent)
    sim.run(0.5)
    ctx = sim.contexts[0]
    first, second = sorted(ctx.orders)
    assert (ctx.orders[first].ahead, ctx.orders[second].ahead) == (5, 6)
    sim.run(1.0)
    assert (ctx.ahead(first), ctx.ahead(second)) == (2, 3)
    sim.run(1.0)
    assert [(f.oid, f.qty, f.ahead, f.left) for f in agent.fills] == [
        (first, 1, 5, 1),
        (second, 1, 6, 1),
    ]


def test_level_swept_through_leaves_nothing():
    agent = Recorder(lambda ctx: ctx.buy(99, 1))
    sim = make([(1.0, market(Side.SELL, 9))], agent)
    sim.run(2.0)
    (f,) = agent.fills
    assert (f.ahead, f.left) == (5, 0)


def test_modify_keeps_the_place_only_when_shrinking_in_place():
    def start(ctx):
        oid = ctx.buy(99, 3)
        ctx.buy(99, 1)
        ctx.modify(oid, 99, 2)
        assert ctx.orders[oid].ahead == 5
        ctx.modify(oid, 99, 4)
        assert ctx.orders[oid].ahead == 6
        ctx.modify(oid, 100, 4)
        assert ctx.orders[oid].ahead == 0

    sim = make([], Recorder(start))
    sim.run(1.0)


def test_taker_fills_have_no_queue():
    agent = Recorder(lambda ctx: ctx.buy(102, 1))
    sim = make([], agent)
    sim.run(1.0)
    assert [(f.maker, f.ahead, f.left) for f in agent.fills] == [(False, None, None)]
