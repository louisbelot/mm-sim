import numpy as np
import pytest
from scripted import ScriptedFlow, limit

from mmsim.book import Side
from mmsim.itch import NS_PER_MINUTE, Msg
from mmsim.micro import Chain, K, Tops, bucket, chain, fit, spreads, tops
from mmsim.sim import Agent, Simulator

B, S = Side.BUY, Side.SELL


def at(minute, kind, ref, side=B, px=0, qty=0, new=0):
    return Msg(round(minute * NS_PER_MINUTE), kind, ref, side, px * 100, qty, new)


def two_state(u, d, stay, swap, r, scale=1000):
    rows = [
        (0, 0, 0, stay),
        (0, 1, 0, swap),
        (0, 0, 2, u * r),
        (0, 1, 2, u * (1 - r)),
        (0, 0, -2, d * r),
        (0, 1, -2, d * (1 - r)),
    ]
    cur, nxt, dm = [], [], []
    for c, x, m, p in rows:
        n = round(p * scale)
        cur += [c, 1 - c] * n
        nxt += [x, 1 - x] * n
        dm += [m, -m] * n
    return chain(np.array(cur), np.array(nxt), np.array(dm), 2)


def kernel(n, seed):
    rng = np.random.default_rng(seed)
    P = rng.random((n, n, 3)) * (rng.random((n, n, 3)) < 0.7)
    P[np.arange(n), np.arange(n), 1] += 0.5
    mirror = n - 1 - np.arange(n)
    P = P + P[mirror][:, mirror][:, :, ::-1]
    return P / P.sum(axis=(1, 2), keepdims=True)


def from_kernel(P):
    n = len(P)
    T = np.zeros((n, len(K)))
    T[:, 0], T[:, 3] = P[:, :, 0].sum(axis=1), P[:, :, 2].sum(axis=1)
    return Chain(np.arange(n), P[:, :, 1], T, P[:, :, 0] + P[:, :, 2])


def walk(P, x0, steps, paths, seed):
    rng = np.random.default_rng(seed)
    n = len(P)
    flat = P.reshape(n, -1).cumsum(axis=1)
    x = np.full(paths, x0)
    m = np.zeros(paths)
    states, moves = [x0], []
    for _ in range(steps):
        k = np.minimum((rng.random(paths)[:, None] > flat[x]).sum(axis=1), 3 * n - 1)
        x, dm = divmod(k, 3)
        m += dm - 1
        if paths == 1:
            states.append(int(x[0]))
            moves.append(int(dm[0]) - 1)
    return m, np.array(states), np.array(moves)


def test_spreads_cover_the_day():
    top = Tops(
        np.array([0.0, 50.0, 90.0, 97.0]),
        np.array([100, 100, 100, 100]),
        np.array([101, 103, 104, 106]),
        np.ones(4, dtype=np.int64),
        np.ones(4, dtype=np.int64),
    )
    assert spreads(top, 100.0) == 4
    assert spreads(top, 100.0, share=0.9) == 3
    assert spreads(top, 100.0, share=0.4) == 2


def test_bucket_is_symmetric_and_centred():
    imb = np.array([0.0, 1 / 11, 0.2, 0.5, 0.8, 10 / 11, 1.0])
    for n in (3, 11):
        assert np.array_equal(bucket(1 - imb, n), n - 1 - bucket(imb, n))
        assert bucket(0.5, n) == n // 2
    assert bucket(0.01, 11) == 0 and bucket(0.99, 11) == 10


def test_first_move_closed_form():
    c = two_state(0.3, 0.1, 0.4, 0.2, 0.5)
    assert c.G1 == pytest.approx([0.2 / 0.8, -0.2 / 0.8])
    assert c.B == pytest.approx(np.full((2, 2), 0.5))
    assert c.gstar() == pytest.approx(c.G1)


def test_persistent_imbalance_closed_form():
    u, d, stay, swap, r = 0.3, 0.1, 0.4, 0.2, 0.7
    c = two_state(u, d, stay, swap, r)
    rho = (u + d) * (2 * r - 1) / (1 - stay + swap)
    g = (u - d) / (1 - stay + swap) / (1 - rho)
    assert c.gstar() == pytest.approx([g, -g])
    assert c.series(200) == pytest.approx(c.gstar())


def test_rows_are_stochastic():
    c = from_kernel(kernel(5, 1))
    assert (c.Q.sum(axis=1) + c.R.sum(axis=1)) == pytest.approx(np.ones(5))
    assert c.T.sum(axis=1) == pytest.approx(c.R.sum(axis=1))
    assert c.B.sum(axis=1) == pytest.approx(np.ones(5))
    assert c.stationary() @ c.G1 == pytest.approx(0, abs=1e-12)


def test_gstar_matches_simulated_long_run_drift():
    P = kernel(5, 2)
    g = from_kernel(P).gstar()
    for x0 in range(5):
        m, _, _ = walk(P, x0, 150, 40_000, 10 + x0)
        se = m.std() / np.sqrt(len(m))
        assert abs(m.mean() - g[x0]) < 4 * se


def test_gstar_is_antisymmetric():
    g = from_kernel(kernel(7, 3)).gstar()
    assert g == pytest.approx(-g[::-1])
    assert g[3] == pytest.approx(0, abs=1e-12)


def synthetic_tops(P, steps, seed):
    _, x, dm = walk(P, 1, steps, 1, seed)
    sizes = np.array([(1, 4), (1, 1), (4, 1)])
    mid2 = 2001 + 2 * np.concatenate([[0], np.cumsum(dm)])
    bid = (mid2 - 1) // 2
    t = np.arange(len(x), dtype=float)
    return Tops(t, bid, bid + 1, sizes[x, 0], sizes[x, 1])


def test_fit_recovers_known_chain():
    P = kernel(3, 4)
    truth = from_kernel(P).gstar()
    m = fit(synthetic_tops(P, 200_000, 5), n_imb=3, n_spread=1)
    assert m.g[0] == pytest.approx(truth, abs=0.03)
    assert m.counts.sum() == 200_000


def test_fit_drops_wide_spreads_and_jumps():
    top = Tops(
        np.arange(6.0),
        np.array([100, 100, 100, 104, 104, 105]),
        np.array([101, 101, 104, 105, 105, 106]),
        np.array([5, 1, 1, 1, 5, 1]),
        np.array([1, 5, 5, 5, 1, 5]),
    )
    m = fit(top, n_imb=3, n_spread=1)
    assert m.counts.sum() == 3
    with pytest.raises(ValueError):
        fit(top, n_imb=4)


def test_adjust_and_price():
    top = Tops(
        np.arange(4.0),
        np.array([100, 101, 101, 101]),
        np.array([101, 102, 102, 102]),
        np.array([9, 1, 9, 5]),
        np.array([1, 9, 1, 5]),
    )
    m = fit(Tops(*(np.tile(a, 50) for a in top)), n_imb=3, n_spread=1)
    assert m.adjust(1, 9, 1) > 0
    assert m.adjust(1, 9, 1) == pytest.approx(-m.adjust(1, 1, 9))
    assert m.adjust(1, 5, 5) == pytest.approx(0, abs=1e-12)
    assert np.isnan(m.adjust(2, 9, 1))
    assert m.price(100, 101, 9, 1) == pytest.approx(100.5 + m.g[0, 2])


def test_tops_follow_the_feed():
    feed = [
        at(0.1, "A", 1, B, 99, 300),
        at(0.2, "A", 2, S, 101, 400),
        at(0.3, "A", 3, B, 99, 200),
        at(0.4, "A", 4, B, 98, 500),
        at(0.5, "E", 1, qty=300),
        at(0.6, "A", 5, S, 100, 100),
        at(0.7, "D", 5),
    ]
    top = tops(feed)
    assert top.t == pytest.approx([0.2, 0.3, 0.5, 0.6, 0.7])
    assert top.bid.tolist() == [99, 99, 99, 99, 99]
    assert top.ask.tolist() == [101, 101, 101, 100, 101]
    assert top.bq.tolist() == [300, 500, 200, 200, 200]
    assert top.aq.tolist() == [400, 400, 400, 100, 400]
    assert top.mid_at(0.65) == 99.5
    assert top.between(0.3, 0.6).t == pytest.approx([0.3, 0.5])
    with pytest.raises(ValueError):
        top.mid_at(0.1)


def test_lean_ignores_own_quotes():
    top = Tops(
        np.arange(4.0),
        np.array([100, 101, 101, 101]),
        np.array([101, 102, 102, 102]),
        np.array([9, 1, 9, 5]),
        np.array([1, 9, 1, 5]),
    )
    m = fit(Tops(*(np.tile(a, 50) for a in top)), n_imb=3, n_spread=1)
    seen = []

    class Probe(Agent):
        def on_start(self, ctx):
            seen.append(m.lean(ctx))
            ctx.buy(99, 40)
            seen.append(m.lean(ctx))
            seen.append(ctx.book.level(B, 99).qty)

    sim = Simulator(ScriptedFlow([(1.0, limit(B, 99, 20))]), [Probe()])
    sim.seed(99, 100)
    sim.run(2.0)
    assert seen == pytest.approx([0.0, 0.0, 45], abs=1e-12)
    assert m.lean(sim.contexts[0]) == pytest.approx(m.g[0, 2])
