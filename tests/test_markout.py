import numpy as np
import pytest

from mmsim.accounting import Account, Fill
from mmsim.book import Side
from mmsim.markout import Tape, curve, markouts, signature


def tape():
    tp = Tape()
    for t, mid in ((0, 100.5), (1, 101), (1, 101.5), (2, 101.5), (3, 100), (4, None)):
        tp.record(t, mid)
    return tp


def fill(t, side, px, qty=1):
    return Fill(t, 0, side, px, qty, True, 0.0, None)


def test_tape_keeps_the_last_mid_of_each_instant():
    tp = tape()
    assert tp.t == [0, 1, 3]
    assert list(tp.at([0, 0.5, 1, 2.9, 3, 10])) == [100.5, 100.5, 101.5, 101.5, 100, 100]
    with pytest.raises(ValueError):
        tp.at(-1)


def test_markouts_by_hand():
    fills = [fill(0.5, Side.BUY, 100), fill(1.0, Side.SELL, 102)]
    m = markouts(fills, tape(), [0, 1, 2.5], end=3.2)
    assert np.array_equal(m, [[0.5, 1.5, 0.0], [0.5, 0.5, np.nan]], equal_nan=True)


def test_curve_weights_by_qty_and_skips_missing():
    fills = [fill(0, Side.BUY, 100, 3), fill(0, Side.BUY, 100, 1)]
    mean, se = curve(fills, np.array([[1.0, 2.0], [-1.0, np.nan]]))
    assert mean == pytest.approx([0.5, 2.0])
    assert se == pytest.approx([np.sqrt(0.75 / 2), 0.0])


def test_markout_to_the_close_is_the_pnl():
    rng = np.random.default_rng(2)
    tp = Tape()
    for t in range(100):
        tp.record(float(t), 100 + rng.integers(-5, 6) / 2)
    fills = [
        fill(float(rng.integers(0, 99)), Side(rng.integers(0, 2)), int(rng.integers(95, 106)))
        for _ in range(300)
    ]
    acct = Account()
    for f in fills:
        acct.fill(f.t, f.oid, f.side, f.price, f.qty, True, None)
    close = markouts(fills, tp, [1000.0])[:, 0]
    assert close.sum() == pytest.approx(acct.value(tp.mid[-1]))


def test_signature_of_a_known_tape():
    tape = Tape()
    for t, mid in enumerate([100.0, 100.5, 101.0, 100.5, 101.0, 101.5, 102.0]):
        tape.record(float(t), mid)
    sig = signature(tape, 0.0, 6.0)
    assert sig.moves == 1.0
    assert sig.eta == pytest.approx(3 / (2 * 2))
    assert sig.vol == pytest.approx(np.std([0.5, 0.5, -0.5, 0.5, 0.5, 0.5]))
