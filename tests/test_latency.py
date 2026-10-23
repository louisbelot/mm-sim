import math
import random

import pytest
from scripted import ScriptedFlow, make, market

from mmsim.book import Side
from mmsim.bots import GLFT
from mmsim.flow import CSTParams, ZeroIntelligence
from mmsim.glft import Model
from mmsim.sim import Agent, Latency, Link, Simulator

SLOW = Latency(entry=0.1, response=0.1)


class Bidder(Agent):
    def __init__(self, px=100, cancel_at=None):
        self.px = px
        self.cancel_at = cancel_at
        self.oid = None
        self.fills = []

    def on_start(self, ctx):
        self.oid = ctx.buy(self.px, 1)
        if self.cancel_at is not None:
            ctx.set_timer(self.cancel_at)

    def on_timer(self, ctx):
        ctx.cancel(self.oid)

    def on_fill(self, ctx, fill):
        self.fills.append((ctx.t, ctx.position, fill))


def test_order_reaches_book_after_entry_latency():
    bot = Bidder()
    sim = make([], bot, latency={bot: Latency(entry=0.1, response=0.2)})
    ctx = sim.contexts[0]
    sim.run(0.05)
    assert bot.oid not in sim.book and not ctx.orders[bot.oid].acked
    sim.run(0.1)
    assert sim.book.best_bid() == 100 and not ctx.orders[bot.oid].acked
    sim.run(0.2)
    assert ctx.orders[bot.oid].acked and ctx.orders[bot.oid].ahead == 0


def test_fill_report_arrives_after_response_latency():
    bot = Bidder()
    lat = Latency(entry=0.1, response=0.25)
    sim = make([(1.0, market(Side.SELL, 1))], bot, latency={bot: lat})
    ctx = sim.contexts[0]
    sim.run(1.1)
    assert ctx.account.position == 1 and ctx.position == 0 and bot.fills == []
    assert bot.oid in ctx.orders
    sim.run(0.2)
    ((t, position, fill),) = bot.fills
    assert t == 1.25 and position == 1 and fill.t == 1.0
    assert ctx.orders == {} and not fill.stale


def test_cancel_that_arrives_late_loses_the_race():
    bot = Bidder(cancel_at=0.95)
    sim = make([(1.0, market(Side.SELL, 1))], bot, latency={bot: SLOW})
    ctx = sim.contexts[0]
    sim.run(2.0)
    ((t, _, fill),) = bot.fills
    assert fill.stale and fill.t == 1.0 and t == 1.1
    assert ctx.orders == {} and ctx.position == 1 and sim.book.find(1).qty == 5
    assert sim._agent_orders == {}


def test_cancel_that_arrives_first_wins():
    bot = Bidder(cancel_at=0.85)
    sim = make([(1.0, market(Side.SELL, 1))], bot, latency={bot: SLOW})
    sim.run(2.0)
    assert bot.fills == [] and sim.contexts[0].orders == {}
    assert sim.book.find(1).qty == 4


def test_cancel_of_an_unacked_order_is_sent_once():
    sent = []

    class Flicker(Agent):
        def on_start(self, ctx):
            oid = ctx.buy(100, 1)
            ctx.cancel(oid)
            ctx.cancel(oid)
            sent.append(len(sim._events))

    bot = Flicker()
    sim = make([], bot, latency={bot: SLOW})
    sim.run(1.0)
    assert sent == [2] and sim.contexts[0].orders == {} and sim.book.best_bid() == 99


def test_rejected_submit_leaves_the_view_after_the_report():
    bot = Bidder(px=0)
    sim = make([], bot, latency={bot: SLOW})
    ctx = sim.contexts[0]
    sim.run(0.15)
    assert bot.oid in ctx.orders
    sim.run(0.1)
    assert ctx.orders == {} and sim._agent_orders == {}


def test_negative_latency_is_an_error():
    with pytest.raises(ValueError):
        Latency(entry=-1e-3)


def test_link_is_fifo_and_adds_its_mean_delay():
    link = Link(0.2, 0.1, random.Random(4))
    burst = [link.arrival(5.0) for _ in range(200)]
    assert burst == sorted(burst) and burst[0] >= 5.2

    link = Link(0.2, 0.1, random.Random(5))
    n = 20_000
    d = [link.arrival(10.0 * i) - 10.0 * i for i in range(1, n + 1)]
    mean = sum(d) / n
    assert min(d) >= 0.2 and abs(mean - 0.3) < 4 * 0.1 / math.sqrt(n)


def test_session_with_latency_settles_consistent():
    bot = GLFT(Model(1.0, 2.1, 3.6, 0.3, 5), keep=1)
    lat = Latency(entry=0.01, response=0.01, jitter=0.01)
    sim = Simulator(ZeroIntelligence(CSTParams.sky_perfect()), [bot], seed=8, latency={bot: lat})
    sim.seed(1000, 1001)
    sim.warm_up(30)
    sim.run(300)
    ctx = sim.contexts[0]
    fills = ctx.account.fills
    assert len(fills) > 50 and any(f.stale for f in fills)

    sim.flow = ScriptedFlow([])
    sim.run(1.0)
    assert sim._events == []
    assert ctx.position == ctx.account.position
    assert sim.book.audit() == ""
    mine = [o for side in Side for o in sim.book.orders(side) if o.owner == ctx.owner]
    resting = {o.id: (o.side, o.price, o.qty) for o in mine}
    assert resting == {oid: (o.side, o.price, o.qty) for oid, o in ctx.orders.items()}
    assert all(o.acked and not o.canceling for o in ctx.orders.values())
