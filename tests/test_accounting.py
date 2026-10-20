import random

import pytest

from mmsim.accounting import Account
from mmsim.book import Side


def test_round_trip_earns_the_spread():
    acct = Account()
    acct.fill(0.0, 1, Side.BUY, 100, 2, True, 100.5)
    acct.fill(1.0, 2, Side.SELL, 102, 2, True, 101.5)
    assert acct.position == 0
    assert acct.cash == 4
    assert acct.value(57.0) == 4
    assert acct.volume == 4


def test_mark_to_market():
    acct = Account()
    acct.fill(0.0, 1, Side.BUY, 100, 3, False, None)
    assert acct.value(101.0) == 3
    assert acct.value(99.5) == -1.5


def test_rebate_and_fee():
    acct = Account(maker_fee=-0.2, taker_fee=0.3)
    maker = acct.fill(0.0, 1, Side.SELL, 100, 5, True, 100.0)
    taker = acct.fill(0.0, 2, Side.BUY, 100, 5, False, 100.0)
    assert maker.fee == pytest.approx(-1.0) and taker.fee == pytest.approx(1.5)
    assert acct.fees == pytest.approx(0.5)
    assert acct.value(100.0) == pytest.approx(-0.5)


def test_value_is_sum_of_fill_markouts():
    rng = random.Random(4)
    acct = Account(maker_fee=-0.1, taker_fee=0.25)
    for i in range(500):
        side = rng.choice((Side.BUY, Side.SELL))
        acct.fill(i, i, side, rng.randint(90, 110), rng.randint(1, 5), rng.random() < 0.7, None)
    mark = 103.25
    markouts = sum(
        (1 if f.side == Side.BUY else -1) * f.qty * (mark - f.price) - f.fee for f in acct.fills
    )
    assert acct.value(mark) == pytest.approx(markouts)
