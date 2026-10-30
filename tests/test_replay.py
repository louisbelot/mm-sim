import random
from pathlib import Path

import pytest

from mmsim.book import Book, EventType, OrderType, Side, Tif
from mmsim.bots import SymmetricQuoter
from mmsim.itch import NS_PER_MINUTE, Msg, read
from mmsim.replay import ItchFlow, ReplayBook, on_grid
from mmsim.sim import Agent, Latency, Simulator

B, S = Side.BUY, Side.SELL


def at(minute, kind, ref, side=B, px=0, qty=0, new=0):
    return Msg(round(minute * NS_PER_MINUTE), kind, ref, side, px * 100, qty, new)


BOOK = [
    at(0.1, "A", 1, B, 99, 300),
    at(0.2, "A", 2, B, 99, 200),
    at(0.3, "A", 3, S, 101, 400),
    at(0.4, "A", 4, B, 98, 500),
]


class Script(Agent):
    def __init__(self, start=None):
        self.start = start
        self.fills = []
        self.prints = []

    def on_start(self, ctx):
        if self.start:
            self.start(ctx)

    def on_fill(self, ctx, fill):
        self.fills.append(fill)

    def on_trade(self, ctx, p):
        self.prints.append(p)

    def on_book(self, ctx):
        pass


def replay(feed, start, until=10.0, **kw):
    agent = Script(start)
    sim = Simulator(ItchFlow(BOOK + feed), [agent], book=ReplayBook, **kw)
    sim.warm_up(1.0)
    sim.run(until - 1.0)
    return sim, agent


def test_book_matches_the_cpp_engine_on_its_own_itch():
    book, traded, value = ReplayBook(), 0, 0
    for m in on_grid(read(Path(__file__).parent / "data" / "session.itch"), 100):
        for e in book.apply(m, (0, 0)):
            traded += e.qty
            value += e.qty * e.price
    assert book.audit() == ""
    assert book.depth(B, 5) == [(98, 200, 1), (97, 100, 1)]
    assert book.depth(S, 5) == [(103, 200, 1), (104, 600, 1)]
    assert (traded, value) == (1200, 120_600)


def real(book):
    return [(o.id, o.price, o.qty) for side in Side for o in book.orders(side) if o.id < 0]


def test_join_behind_and_fill_when_the_order_behind_executes():
    feed = [
        at(2, "A", 5, B, 99, 100),
        at(3, "E", 1, qty=300),
        at(4, "E", 2, qty=150),
        at(5, "E", 2, qty=50),
        at(5, "E", 5, qty=100),
    ]
    sim, agent = replay(feed, lambda ctx: ctx.buy(99, 80))
    assert [(f.t, f.price, f.qty, f.ahead) for f in agent.fills] == [(5.0, 99, 80, 500)]
    assert real(sim.book) == [(-4, 98, 500), (-3, 101, 400)]


def test_ahead_shrinks_with_cancels_and_executions_in_front():
    seen = []

    class Watch(Script):
        def on_book(self, ctx):
            for oid in ctx.orders:
                seen.append(ctx.ahead(oid))

    agent = Watch(lambda ctx: ctx.buy(99, 10))
    feed = [at(2, "X", 1, qty=100), at(3, "E", 1, qty=50), at(4, "D", 2), at(5, "E", 1, qty=150)]
    sim = Simulator(ItchFlow(BOOK + feed), [agent], book=ReplayBook)
    sim.warm_up(1.0)
    sim.run(9.0)
    assert seen == [400, 350, 150, 0]
    assert agent.fills == []


def test_better_price_fills_first_but_the_real_book_does_not_move():
    feed = [at(2, "E", 1, qty=120)]
    sim, agent = replay(feed, lambda ctx: ctx.buy(100, 50))
    assert [(f.price, f.qty, f.maker) for f in agent.fills] == [(100, 50, True)]
    assert real(sim.book)[0] == (-1, 99, 180)
    assert [(p.side, p.price, p.qty) for p in agent.prints] == [(S, 99, 170)]


def test_real_add_through_a_quote_fills_it_and_rests_in_full():
    feed = [at(2, "A", 7, B, 100, 30)]
    sim, agent = replay(feed, lambda ctx: ctx.sell(100, 50))
    assert [(f.side, f.price, f.qty) for f in agent.fills] == [(S, 100, 30)]
    assert sim.book.find(-7).qty == 30
    assert sim.book.best_bid() == 100 and sim.book.best_ask() == 101
    assert sim.contexts[0].orders.popitem()[1].qty == 20


def test_replace_loses_priority_to_the_bot():
    feed = [at(2, "U", 1, px=99, qty=300, new=8), at(3, "E", 2, qty=200), at(4, "E", 8, qty=40)]
    sim, agent = replay(feed, lambda ctx: ctx.buy(99, 25))
    assert [(f.t, f.qty) for f in agent.fills] == [(4.0, 25)]
    assert sim.book.find(-8).qty == 260


def test_bot_takes_liquidity_without_removing_it():
    def take(ctx):
        ctx.buy(101, 150, tif=Tif.IOC)
        ctx.submit(B, 0, 500, kind=OrderType.MARKET)
        ctx.buy(101, 600)

    sim, agent = replay([], take)
    assert [(f.price, f.qty, f.maker) for f in agent.fills] == [
        (101, 150, False),
        (101, 400, False),
        (101, 400, False),
    ]
    assert sim.book.find(-3).qty == 400
    assert sim.contexts[0].orders == {} and sim.contexts[0].position == 950


def test_resting_bot_order_is_taken_for_real_by_another_bot():
    a = Script(lambda ctx: ctx.sell(100, 10))
    b = Script(lambda ctx: ctx.buy(101, 30, tif=Tif.IOC))
    sim = Simulator(ItchFlow(BOOK), [a, b], book=ReplayBook)
    sim.warm_up(1.0)
    sim.run(1.0)
    assert [f.qty for f in a.fills] == [10] and [f.qty for f in b.fills] == [10, 20]
    assert sim.book.find(-3).qty == 400 and len(sim.book) == 4


def test_post_only_inside_the_spread_and_the_real_mid():
    sim, agent = replay([], lambda ctx: ctx.buy(100, 10, post_only=True))
    ctx = sim.contexts[0]
    assert len(ctx.orders) == 1
    assert sim.book.best_bid() == 99 and sim.book.mid() == 100.0
    assert sim.book.audit() == ""


def test_sweep_is_one_print():
    feed = [at(2, "E", 1, qty=300), at(2, "E", 2, qty=200), at(2, "E", 4, qty=10)]
    _, agent = replay(feed, None)
    assert [(p.t, p.side, p.price, p.qty) for p in agent.prints] == [(2.0, S, 98, 510)]


def test_off_grid_orders_are_dropped_with_their_history():
    msgs = [
        Msg(0, "A", 1, B, 1, 100),
        Msg(1, "U", 1, px=9900, qty=100, new=2),
        Msg(2, "A", 3, S, 10100, 50),
        Msg(3, "U", 3, px=10150, qty=50, new=4),
        Msg(4, "E", 4, qty=10),
        Msg(5, "D", 4),
        Msg(6, "A", 5, B, 3, 1),
        Msg(7, "E", 5, qty=1),
    ]
    out = list(on_grid(msgs, 100))
    assert out == [
        Msg(1, "A", 2, B, 99, 100),
        Msg(2, "A", 3, S, 101, 50),
        Msg(3, "D", 3),
    ]


def random_feed(n, rng):
    book, ids, msgs, t = Book(), iter(range(1, 10**9)), [], 0
    for _ in range(n):
        t += rng.randrange(1, 4) * NS_PER_MINUTE // 1000
        bid, ask = book.best_bid() or 100, book.best_ask() or 102
        roll = rng.random()
        live = [o for side in Side for o in book.orders(side)]
        if roll < 0.45 or len(live) < 4:
            side = rng.choice(list(Side))
            px = ask - rng.randint(1, 4) if side == B else bid + rng.randint(1, 4)
            ref, qty = next(ids), rng.randint(1, 6) * 100
            book.submit(ref, side, px, qty)
            msgs.append(Msg(t, "A", ref, side, px * 100, qty))
        elif roll < 0.6:
            for e in book.submit(
                next(ids), rng.choice(list(Side)), 0, rng.randint(50, 900), kind=OrderType.MARKET
            ):
                if e.type == EventType.TRADE:
                    msgs.append(Msg(t, "E", e.maker, qty=e.qty))
        elif roll < 0.75:
            o = rng.choice(live)
            if o.qty > 1:
                q = rng.randint(1, o.qty - 1)
                book.modify(o.id, o.price, o.qty - q)
                msgs.append(Msg(t, "X", o.id, qty=q))
        elif roll < 0.85:
            o = rng.choice(live)
            ref = next(ids)
            book.cancel(o.id)
            px = o.price + rng.choice([-1, 0, 1])
            if (o.side == B and px < (book.best_ask() or 10**9)) or (
                o.side == S and px > (book.best_bid() or 0)
            ):
                book.submit(ref, o.side, px, o.qty)
                msgs.append(Msg(t, "U", o.id, px=px * 100, qty=o.qty, new=ref))
            else:
                msgs.append(Msg(t, "D", o.id))
        else:
            o = rng.choice(live)
            book.cancel(o.id)
            msgs.append(Msg(t, "D", o.id))
    return book, msgs


@pytest.mark.parametrize("seed", range(12))
def test_the_bot_never_moves_the_real_book(seed):
    rng = random.Random(seed)
    truth, msgs = random_feed(3000, rng)
    bot = SymmetricQuoter(0.5, size=rng.choice([10, 100, 300]), max_position=900)
    sim = Simulator(ItchFlow(msgs), [bot], book=ReplayBook)
    sim.warm_up(msgs[200].t / NS_PER_MINUTE)
    sim.run(msgs[-1].t / NS_PER_MINUTE - sim.t + 1)
    want = [(-o.id, o.price, o.qty) for side in Side for o in truth.orders(side)]
    assert real(sim.book) == want
    assert sim.book.audit() == ""
    assert len(sim.contexts[0].account.fills) > 20


@pytest.mark.parametrize("seed", range(4))
def test_the_feed_replica_ends_on_the_real_book(seed):
    rng = random.Random(seed)
    _, msgs = random_feed(2000, rng)
    bot = SymmetricQuoter(0.5, size=100, max_position=900)
    lag = {bot: Latency(feed=1e-4, entry=1e-4, response=1e-4)}
    sim = Simulator(ItchFlow(msgs), [bot], book=ReplayBook, latency=lag)
    sim.warm_up(msgs[200].t / NS_PER_MINUTE)
    sim.run(msgs[-1].t / NS_PER_MINUTE - sim.t + 1)
    view = sim.contexts[0]._view
    assert real(view) == real(sim.book)
    assert view.audit() == "" and sim.book.audit() == ""


def test_latency_loses_the_queue_position():
    feed = [at(2.0, "A", 5, B, 99, 100), at(3.0, "E", 1, qty=300)]
    feed += [at(3.1, "E", 2, qty=200), at(3.2, "E", 5, qty=100)]

    def fills(delay):
        def start(ctx):
            ctx.buy(99, 10)

        agent = Script(start)
        lag = {agent: Latency(entry=delay)} if delay else None
        sim = Simulator(ItchFlow(BOOK + feed), [agent], book=ReplayBook, latency=lag)
        sim.warm_up(1.9)
        sim.run(8.0)
        return agent.fills

    assert fills(0.0)[0].t == pytest.approx(3.2)
    assert fills(0.0)[0].ahead == 500
    assert fills(0.5) == []


def test_a_rebate_is_paid_on_replayed_fills():
    feed = [at(2.0, "A", 5, B, 99, 100), at(3.0, "E", 1, qty=300), at(3.1, "E", 2, qty=200)]
    feed += [at(3.2, "E", 5, qty=100)]
    sim, agent = replay(feed, lambda ctx: ctx.buy(99, 10), maker_fee=-0.2)
    assert [f.fee for f in agent.fills] == [-2.0]
    assert sim.contexts[0].account.fees == -2.0
