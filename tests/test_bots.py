import math
from types import SimpleNamespace

from scripted import limit, make, market

from mmsim.book import Side
from mmsim.bots import GLFT, AvellanedaStoikov, SymmetricQuoter
from mmsim.flow import CSTParams, ZeroIntelligence
from mmsim.glft import Model
from mmsim.sim import Simulator


def quotes(ctx):
    return sorted((o.side, o.price) for o in ctx.orders.values())


def test_symmetric_quoter_joins_the_inside():
    sim = make([], SymmetricQuoter())
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

    sim.run(1.0)
    assert bot._oids == first

    sim.run(1.0)
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


def as_targets(position, t=0.0, horizon=None, gamma=0.1):
    ctx = SimpleNamespace(mid=100.5, bid=99, ask=102, position=position, t=t, end=10.0)
    return AvellanedaStoikov(gamma, 1.0, 3.6, horizon=horizon).targets(ctx)


def test_as_quotes_by_hand():
    assert as_targets(0) == (99, 102)
    assert as_targets(1) == (98, 101)
    assert as_targets(5) == (94, 100)
    assert as_targets(-5) == (101, 107)


def test_as_spread_narrows_near_the_end():
    assert as_targets(0, t=9.9) == (100, 101)
    assert as_targets(0, horizon=0.1) == (100, 101)


def test_as_bot_leans_against_its_inventory():
    bot = AvellanedaStoikov(0.5, 1.0, 3.6)
    script = [(0.5, limit(Side.BUY, 97)), (1.0, market(Side.SELL, 6))]
    sim = make(script, bot)
    sim.run(2.0)
    ctx = sim.contexts[0]
    assert ctx.position == 1
    assert quotes(ctx) == [(Side.BUY, 98), (Side.SELL, 100)]


BOOK = Model(sigma=1.0, A=2.1, k=3.6, gamma=1.0, Q=3)


def glft_targets(position, size=1):
    ctx = SimpleNamespace(mid=100.5, bid=99, ask=102, position=position)
    return GLFT(BOOK, size=size).targets(ctx)


def test_glft_quotes_by_hand():
    bids, asks = BOOK.quotes(float("inf"))
    assert glft_targets(0) == (99, 102)
    assert glft_targets(2) == (math.floor(100.5 - bids[5]), 101)
    assert glft_targets(-2) == (100, math.ceil(100.5 + asks[1]))
    assert glft_targets(3)[0] is None and glft_targets(-3)[1] is None
    assert glft_targets(4, size=2) == glft_targets(2)


def test_glft_on_cst_flow_respects_its_bound():
    bot = GLFT(BOOK)
    sim = Simulator(ZeroIntelligence(CSTParams.sky_perfect()), [bot], seed=3)
    sim.seed(1000, 1001)
    sim.warm_up(30)
    sim.run(390)
    ctx = sim.contexts[0]
    assert sim.book.audit() == ""
    assert max(abs(s.position) for s in ctx.history) <= 3
    assert len(ctx.account.fills) > 50
