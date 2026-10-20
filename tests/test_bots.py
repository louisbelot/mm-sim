from types import SimpleNamespace

from scripted import limit, make, market

from mmsim.book import Side
from mmsim.bots import AvellanedaStoikov, SymmetricQuoter
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


def as_targets(position, t=0.0, horizon=None, gamma=0.1):
    """AS targets with others at 99 / 102 (mid 100.5), sigma = 1, k = 3.6, run ending at 10."""
    ctx = SimpleNamespace(mid=100.5, bid=99, ask=102, position=position, t=t, end=10.0)
    return AvellanedaStoikov(gamma, 1.0, 3.6, horizon=horizon).targets(ctx)


def test_as_quotes_by_hand():
    # tau = 10: spread = 0.1 * 10 + 20 ln(1 + 0.1 / 3.6) = 1.548, half 0.774.
    assert as_targets(0) == (99, 102)  # 99.73 and 101.27, rounded away from r
    # long one lot: r = 100.5 - 1 = 99.5, quotes 98.73 and 100.27
    assert as_targets(1) == (98, 101)
    # long five: r = 95.5 puts the ask at 96.27, kept one tick above the bid at 99
    assert as_targets(5) == (94, 100)
    assert as_targets(-5) == (101, 107)


def test_as_spread_narrows_near_the_end():
    # tau = 0.1: spread = 0.01 + 0.548, so the quotes fit inside 99 / 102
    assert as_targets(0, t=9.9) == (100, 101)
    assert as_targets(0, horizon=0.1) == (100, 101)


def test_as_bot_leans_against_its_inventory():
    bot = AvellanedaStoikov(0.5, 1.0, 3.6)
    script = [(0.5, limit(Side.BUY, 97)), (1.0, market(Side.SELL, 6))]
    sim = make(script, bot)  # others 99 / 102; the bot joins at 99 behind 5 lots
    sim.run(2.0)
    ctx = sim.contexts[0]
    # The sell takes the 5 lots and the bot's lot. At t = 1 (tau = 1), others are
    # 97 / 102 and r = 99.5 - 0.5 = 99: quotes 98.49 and 99.51. Flat, the ask
    # would have been 101.
    assert ctx.position == 1
    assert quotes(ctx) == [(Side.BUY, 98), (Side.SELL, 100)]
