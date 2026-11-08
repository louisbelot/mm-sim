import numpy as np
import pytest

from mmsim.micro import Tops
from mmsim.ofi import Window, bins, events, flow, integral, oos_r2, regress, scaling


def book(bid, ask, bq, aq, t=None):
    n = len(bid)
    t = np.arange(n, dtype=float) if t is None else np.asarray(t, dtype=float)
    return Tops(t, *(np.asarray(a, dtype=np.int64) for a in (bid, ask, bq, aq)))


def random_book(n, seed):
    rng = np.random.default_rng(seed)
    bid = 100 + np.cumsum(rng.integers(-1, 2, n))
    ask = bid + rng.integers(1, 3, n)
    return book(bid, ask, rng.integers(1, 50, n), rng.integers(1, 50, n), np.cumsum(rng.random(n)))


def test_each_case_of_the_definition():
    top = book(
        [100, 101, 101, 100, 100, 100, 100],
        [102, 102, 102, 102, 103, 101, 101],
        [5, 7, 3, 9, 9, 9, 9],
        [4, 4, 4, 4, 6, 2, 8],
    )
    assert events(top).tolist() == [7, -4, -3, 4, -2, -6]


def test_flow_telescopes_when_prices_hold():
    rng = np.random.default_rng(0)
    bq, aq = rng.integers(1, 99, 200), rng.integers(1, 99, 200)
    top = book(np.full(200, 100), np.full(200, 101), bq, aq)
    i = np.array([0, 17, 120, 199])
    got = flow(top, top.t[i])
    assert np.diff(got).tolist() == (np.diff(bq[i]) - np.diff(aq[i])).tolist()


def test_mirrored_book_has_opposite_flow():
    top = random_book(500, 1)
    flip = Tops(top.t, -top.ask, -top.bid, top.aq, top.bq)
    assert events(flip).tolist() == (-events(top)).tolist()


def test_flow_counts_events_at_the_edge():
    top = book([100, 101, 101], [102, 102, 102], [5, 7, 3], [4, 4, 4])
    assert flow(top, [0.0, 0.5, 1.0, 1.5, 2.0]).tolist() == [0, 0, 7, 7, 3]


def test_integral_matches_brute_force():
    top = random_book(60, 2)
    v = top.bq.astype(float)
    grid = np.linspace(top.t[0], top.t[-1] + 1, 200_001)
    brute = np.concatenate([[0], np.cumsum(v[top.index(grid[:-1])] * np.diff(grid))])
    for k in (0, 1000, 77_777, 200_000):
        assert integral(top, v, grid[k]) == pytest.approx(brute[k], rel=1e-3, abs=1e-3)


def test_bins_add_up():
    top = random_book(3000, 3)
    b = bins(top, 10.0, 1010.0, 50.0)
    assert len(b.t) == 20
    assert b.dp.sum() == pytest.approx(top.mid_at(1010.0) - top.mid_at(10.0))
    assert b.ofi.sum() == flow(top, 1010.0) - flow(top, 10.0)
    flat = book(np.full(5, 100), np.full(5, 101), np.full(5, 30), np.full(5, 10))
    assert bins(flat, 0.0, 4.0, 2.0).depth.tolist() == [20.0, 20.0]


def test_regress_recovers_a_line():
    x = np.linspace(-5, 5, 50)
    f = regress(x, 0.3 + 2 * x)
    assert (f.alpha, f.beta, f.r2) == pytest.approx((0.3, 2.0, 1.0))
    assert f.predict([1.0]) == pytest.approx([2.3])


def test_white_t_matches_classic_when_homoskedastic():
    rng = np.random.default_rng(4)
    x = rng.normal(size=20_000)
    y = 0.05 * x + rng.normal(size=x.size)
    f = regress(x, y)
    classic = f.beta / (np.std(y - f.predict(x), ddof=2) / np.sqrt(((x - x.mean()) ** 2).sum()))
    assert f.t == pytest.approx(classic, rel=0.02)


def test_white_t_widens_under_heteroskedasticity():
    rng = np.random.default_rng(5)
    x = rng.normal(size=20_000)
    y = 0.05 * x + np.abs(x) * 3 * rng.normal(size=x.size)
    f = regress(x, y)
    classic = f.beta / (np.std(y - f.predict(x), ddof=2) / np.sqrt(((x - x.mean()) ** 2).sum()))
    assert abs(f.t) < 0.7 * abs(classic)


def test_clusters_widen_for_repeated_rows():
    rng = np.random.default_rng(6)
    x = rng.normal(size=400)
    y = 0.1 * x + rng.normal(size=400)
    once = regress(x, y)
    rep = regress(np.repeat(x, 9), np.repeat(y, 9), groups=np.repeat(np.arange(400), 9))
    assert rep.beta == pytest.approx(once.beta)
    assert rep.t == pytest.approx(once.t)
    assert regress(x, y, groups=np.arange(400)).t == pytest.approx(once.t)


def test_oos_r2_is_against_no_change():
    assert oos_r2([1.0, -1.0], [1.0, -1.0]) == 1.0
    assert oos_r2([1.0, -1.0], [0.0, 0.0]) == 0.0
    assert oos_r2([1.0, -1.0], [-1.0, 1.0]) == -3.0


def test_scaling_recovers_inverse_depth():
    depth = np.array([200.0, 500.0, 1000.0, 3000.0])
    ws = [Window(0.0, regress([0.0, 1.0, 2.0], [0.0, 0.4 / d, 0.8 / d]), d) for d in depth]
    f = scaling(ws)
    assert f.beta == pytest.approx(-1.0)
    assert np.exp(f.alpha) == pytest.approx(0.4)
