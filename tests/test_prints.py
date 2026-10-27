from scripted import limit, make, market

from mmsim.book import OrderType, Side
from mmsim.bots import GLFT
from mmsim.flow import CSTParams, HawkesFlow
from mmsim.glft import Model
from mmsim.hawkes import Hawkes
from mmsim.sim import Agent, Latency, Print, Simulator


class Tape(Agent):
    def __init__(self, bid=None):
        self.bid = bid
        self.log = []

    def on_start(self, ctx):
        if self.bid is not None:
            ctx.buy(self.bid, 1)

    def on_trade(self, ctx, p):
        self.log.append(("trade", ctx.t, p))

    def on_fill(self, ctx, fill):
        self.log.append(("fill", ctx.t, fill.price))

    def on_book(self, ctx):
        self.log.append(("book", ctx.t))

    def prints(self):
        return [x[2] for x in self.log if x[0] == "trade"]


class Lifter(Agent):
    def on_start(self, ctx):
        ctx.submit(Side.BUY, 0, 1, kind=OrderType.MARKET)


SCRIPT = [
    (0.5, limit(Side.BUY, 98, 3)),
    (1.0, market(Side.SELL, 7)),
    (2.0, market(Side.BUY, 2)),
]


def test_one_print_per_aggressive_order_at_its_worst_price():
    bot = Tape()
    sim = make(SCRIPT, bot)
    sim.run(3.0)
    assert bot.prints() == [Print(1.0, Side.SELL, 98, 7), Print(2.0, Side.BUY, 102, 2)]


def test_print_comes_before_the_fill_and_the_book_update():
    bot = Tape(bid=99)
    sim = make([(1.0, market(Side.SELL, 6))], bot)
    sim.run(2.0)
    kinds = [x[0] for x in bot.log if x[1] == 1.0]
    assert kinds == ["trade", "fill", "book"]


def test_lagged_feed_delivers_prints_late_with_exchange_time():
    bot = Tape()
    sim = make(SCRIPT, bot, latency={bot: Latency(feed=0.3)})
    sim.run(3.0)
    trades = [(x[1], x[2]) for x in bot.log if x[0] == "trade"]
    assert trades == [
        (1.3, Print(1.0, Side.SELL, 98, 7)),
        (2.3, Print(2.0, Side.BUY, 102, 2)),
    ]


def test_other_agents_trades_are_on_the_tape():
    bot = Tape()
    sim = make([], bot, Lifter())
    sim.run(1.0)
    assert bot.prints() == [Print(0.0, Side.BUY, 102, 1)]


def test_no_prints_during_warm_up():
    bot = Tape()
    sim = make([(1.0, market(Side.SELL, 1))], bot)
    sim.warm_up(2.0)
    sim.run(1.0)
    assert bot.prints() == []
    assert sim.prints == [Print(1.0, Side.SELL, 99, 1)]


class Mids(Agent):
    def __init__(self):
        self.seen = []

    def on_trade(self, ctx, p):
        self.seen.append((p, ctx.book.mid()))


def test_simulator_prints_are_on_its_clock():
    watcher = Mids()
    lagged = GLFT(Model(1.0, 2.1, 3.6, 0.3, 5), keep=1)
    flow = HawkesFlow(CSTParams.sky_perfect(), Hawkes.symmetric(0.94, 0.6, 0.1, 6.0))
    lat = {lagged: Latency(0.01, 0.01, 0.01, jitter=0.005)}
    sim = Simulator(flow, [watcher, lagged], seed=3, latency=lat)
    sim.seed(10_000, 10_001)
    sim.warm_up(30)
    sim.run(120)
    live = [p for p in sim.prints if p.t >= 30]
    assert len(sim.prints) > len(live) > 100
    assert [p for p, _ in watcher.seen] == live
    assert all(mid is None or sim.tape.at(p.t) == mid for p, mid in watcher.seen)
