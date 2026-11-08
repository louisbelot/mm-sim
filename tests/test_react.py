from mmsim import qr
from mmsim.book import Side
from mmsim.itch import NS_PER_MINUTE, Msg
from mmsim.qr import CANCEL, LIMIT

B, S = Side.BUY, Side.SELL


def at(minute, *args, **kw):
    return Msg(round(minute * NS_PER_MINUTE), *args, **kw)


def test_quotes_are_the_touch_events():
    msgs = [
        at(0, "A", 1, B, 10_000, 3),
        at(0, "A", 2, B, 9_900, 2),
        at(0, "A", 3, S, 10_100, 4),
        at(0, "A", 4, S, 10_200, 1),
        at(1, "D", 1),
        at(2, "A", 5, B, 9_800, 1),
        at(3, "A", 6, B, 9_900, 2),
        at(4, "D", 4),
        at(5, "A", 7, S, 10_100, 2),
        at(6, "E", 3, qty=1),
    ]
    f = qr.fit(msgs, 0.0, 10.0, depth=3, bins=4, aes=[1, 1, 1], edges=(1, 2))
    got = [(f.t[r], s, k) for r, s, k in f.quotes]
    assert got == [(1.0, B, CANCEL), (3.0, B, LIMIT), (5.0, S, LIMIT)]
    assert [(f.t[r], s) for r, s in f.hits] == [(6.0, S)]
