import numpy as np
import pytest

from mmsim import qr
from mmsim.book import Side
from mmsim.itch import NS_PER_MINUTE, Msg
from mmsim.qr import CANCEL, LIMIT, MARKET, Gap, Params, invariant

B, S = Side.BUY, Side.SELL


def sec(s, kind, ref, side=B, px=0, qty=0):
    return Msg(round(s * NS_PER_MINUTE / 60), kind, ref, side, px * 100, qty)


def pre_open():
    return [
        sec(1, "A", 1, B, 99, 100),
        sec(2, "A", 2, B, 99, 300),
        sec(3, "A", 3, S, 100, 200),
        sec(4, "A", 4, B, 98, 50),
        sec(5, "A", 6, S, 101, 500),
    ]


def test_hand_worked_fit():
    msgs = [
        *pre_open(),
        sec(70, "A", 5, B, 99, 100),
        sec(75, "A", 8, B, 98, 40),
        sec(80, "D", 2),
        sec(90, "E", 3, qty=200),
        sec(100, "A", 7, B, 100, 100),
        sec(110, "D", 6),
    ]
    f = qr.fit(msgs, 1.0, 3.0, depth=2, bins=10)
    assert f.aes.tolist() == [240, 40]
    lvl, n, kind, qty, regime, pos = f.events.T
    assert lvl.tolist() == [0, 1, 0, 0, 0, 0]
    assert kind.tolist() == [LIMIT, LIMIT, CANCEL, MARKET, LIMIT, CANCEL]
    assert qty.tolist() == [100, 40, 300, -200, 100, -500]
    assert n.tolist() == [2, 2, 3, 1, 0, 3]
    assert pos.tolist() == [-1, -1, 500, -1, -1, -1]
    assert regime[4] == qr.STAYED_MARKET + qr.BASE
    assert f.tau[0].sum() == pytest.approx(2 * 50 / 60)
    assert f.tau[1].sum() == pytest.approx(2 * 50 / 60)
    assert f.counts.sum() == 6


def test_gap_names_the_side_that_left():
    g = Gap()
    g.update(100, 101, False)
    assert not g.wide
    g.update(100, 102, True)
    assert g.side == S and g.market and g.up
    assert g.regime(S) == qr.LEFT_MARKET and g.regime(B) == qr.STAYED_MARKET
    g.update(101, 102, False)
    assert not g.wide and g.up
    g.update(100, 102, False)
    assert g.side == B and not g.market and not g.up
    assert g.regime(B) == qr.LEFT_CANCEL and g.regime(S) == qr.STAYED_CANCEL


def flat(K, N, birth, cancel, market, aes=1, sizes=None, **kw):
    table = tuple(
        (float(birth(n)), float(cancel(n)) if n else 0.0, float(market(n)) if n else 0.0)
        for n in range(N + 1)
    )
    rates = tuple(tuple(table for _ in range(qr.REGIMES)) for _ in range(K))
    if sizes is None:
        sizes = {(k, i): (aes,) for k in (LIMIT, CANCEL, MARKET) for i in range(K)}
    queues = tuple((3 * aes,) for _ in range(K))
    return Params(rates, sizes, (aes,) * K, (2 * aes, 4 * aes), queues, False, **kw)


def test_invariant_closed_form():
    birth = np.array([2.0, 2.0, 2.0, 0.0])
    death = np.array([0.0, 1.0, 1.0, 1.0])
    assert invariant(birth, death) == pytest.approx(np.array([1, 2, 4, 8]) / 15)
    assert invariant(np.array([1.0, 3.0, 0.0]), np.array([0, 2.0, 6.0])) == pytest.approx(
        np.array([1, 0.5, 0.25]) / 1.75
    )


def test_fit_keeps_where_cancels_came_from():
    msgs = [
        *pre_open(),
        sec(61, "A", 10, B, 99, 100),
        sec(62, "A", 11, B, 99, 100),
        sec(63, "D", 11),
        sec(64, "D", 10),
        sec(65, "D", 1),
        sec(70, "A", 12, B, 98, 10),
        sec(71, "A", 13, S, 102, 10),
    ]
    f = qr.fit(msgs, 1.0, 3.0, depth=2, bins=10)
    assert f.params(min_events=1).back[0] == (1.0, 1.0, 0.0)


def test_rates_reflect_at_the_top_and_fall_back_when_thin():
    n = np.array([[30.0, 0, 0], [0, 0, 0], [0, 0, 0]])
    pooled_n = np.array([[100.0, 0, 0], [0, 50, 50], [10, 40, 0]])
    rates = qr._filled(n, np.ones(3), pooled_n, np.array([2.0, 5.0, 10.0]), 30)
    assert rates == ((30.0, 0.0, 0.0), (0.0, 10.0, 10.0), (0.0, 4.0, 0.0))


def test_unit_sizes_are_one_aes():
    msgs = [*pre_open(), sec(70, "A", 5, B, 99, 100), sec(75, "A", 8, B, 98, 40)]
    p = qr.fit(msgs, 1.0, 3.0, depth=2).params(sizes=qr.UNIT, min_events=1)
    assert p.sizes[LIMIT, 0] == (100,) and p.sizes[LIMIT, 1] == (40,)
    assert not p.conditioned


