import pytest

from mmsim.accounting import Account
from mmsim.attribution import Attribution, attribute
from mmsim.book import Side
from mmsim.bots import GLFT
from mmsim.flow import CSTParams, ZeroIntelligence
from mmsim.glft import Model
from mmsim.markout import Tape, markouts
from mmsim.sim import Simulator


def hand_tape():
    tape = Tape()
    for t, mid in ((0.0, 99.5), (1.0, 99.0), (1.5, 98.5), (2.0, 100.5), (3.0, 100.0)):
        tape.record(t, mid)
    return tape


def hand_fills():
    acct = Account()
    acct.fill(1.0, 1, Side.BUY, 99, 2, True, 99.5)
    acct.fill(2.5, 2, Side.SELL, 101, 1, True, 100.5)
    return acct


def test_hand_worked_split():
    a = attribute(hand_fills().fills, hand_tape(), 1.0, 3.0)
    assert a == Attribution(spread=1.5, adverse=2.5, inventory=-1.0, fees=0.0)
    assert a.total == hand_fills().value(100.0) == 3.0


def test_spread_plus_adverse_is_the_markout():
    fills = hand_fills().fills
    m = markouts(fills, hand_tape(), [0.5], 3.0)[:, 0]
    a = attribute(fills, hand_tape(), 0.5, 3.0)
    assert a.spread + a.adverse == pytest.approx(2 * m[0] + m[1])
    assert a.adverse == -1.5


def test_horizon_past_the_close_leaves_no_inventory_term():
    a = attribute(hand_fills().fills, hand_tape(), 10.0, 3.0)
    assert a.inventory == 0 and a.spread + a.adverse == 3.0


def test_one_sided_book_gets_no_spread_credit():
    acct = Account()
    acct.fill(1.0, 1, Side.BUY, 99, 1, False, None)
    a = attribute(acct.fills, hand_tape(), 1.0, 3.0)
    assert a.spread == 0 and a.total == 1.0


def test_no_fills():
    assert attribute([], hand_tape(), 1.0, 3.0).total == 0


@pytest.mark.parametrize("horizon", [0.0, 0.5, 5.0])
def test_terms_add_up_to_the_account_in_a_session(horizon):
    bot = GLFT(Model(1.0, 2.1, 3.6, 0.3, 5), keep=1)
    sim = Simulator(ZeroIntelligence(CSTParams.sky_perfect()), [bot], seed=4, maker_fee=-0.2)
    sim.seed(10_000, 10_001)
    sim.warm_up(30)
    sim.run(120)
    acct = sim.contexts[0].account
    a = attribute(acct.fills, sim.tape, horizon, sim.end)
    assert len(acct.fills) > 30 and a.fees < 0 and a.spread > 0
    assert a.total == pytest.approx(acct.value(float(sim.tape.at(sim.end))), abs=1e-9)
