from scripted import limit, make, market

from mmsim.book import Side
from mmsim.bots import SymmetricQuoter
from mmsim.flow import CSTParams, ZeroIntelligence
from mmsim.sim import Simulator


def quotes(ctx):
    return sorted((o.side, o.price) for o in ctx.orders.values())


def test_symmetric_quoter_joins_the_inside():
    sim = make([], SymmetricQuoter())  # others: bid 99, ask 102, mid 100.5
    sim.run(1.0)
    assert quotes(sim.contexts[0]) == [(Side.BUY, 100), (Side.SELL, 101)]


def test_wider_half_spread_rounds_away_from_mid():
    sim = make([], SymmetricQuoter(half_spread=1.2))
    sim.run(1.0)
    assert quotes(sim.contexts[0]) == [(Side.BUY, 99), (Side.SELL, 102)]


def test_requotes_when_the_market_moves_and_keeps_queue_otherwise():
    bot = SymmetricQuoter()
    sim = make([(1.0, limit(Side.SELL, 104)), (2.0, limit(Side.BUY, 100))], bot)
    sim.run(0.5)
    ctx = sim.contexts[0]
    first = dict(bot._oids)

    sim.run(1.0)  # an ask behind the best changes nothing
    assert bot._oids == first

    sim.run(1.0)  # others now 100 / 102, mid 101: only the ask moves
    assert quotes(ctx) == [(Side.BUY, 100), (Side.SELL, 102)]
    assert bot._oids[Side.BUY] == first[Side.BUY]
    assert bot._oids[Side.SELL] != first[Side.SELL]
    assert len(ctx.orders) == 2


def test_requotes_after_a_fill():
    bot = SymmetricQuoter()
    sim = make([(1.0, market(Side.SELL, 1))], bot)
    sim.run(2.0)
    ctx = sim.contexts[0]
    assert ctx.position == 1
    assert quotes(ctx) == [(Side.BUY, 100), (Side.SELL, 101)]


def test_position_limit_drops_a_side():
    bot = SymmetricQuoter(max_position=2)
    sim = make([(1.0, market(Side.SELL, 1)), (2.0, market(Side.SELL, 1))], bot)
    sim.run(3.0)
    ctx = sim.contexts[0]
    assert ctx.position == 2
    assert quotes(ctx) == [(Side.SELL, 101)]


def test_quoter_on_cst_flow_stays_within_limits():
    bot = SymmetricQuoter(max_position=5)
    sim = Simulator(ZeroIntelligence(CSTParams.sky_perfect()), [bot], seed=8)
    sim.seed(1000, 1001)
    sim.warm_up(30)
    sim.run(390)
    ctx = sim.contexts[0]
    assert sim.book.audit() == ""
    assert max(abs(s.position) for s in ctx.history) <= 5
    assert len(ctx.account.fills) > 50
    assert all(f.maker for f in ctx.account.fills)
