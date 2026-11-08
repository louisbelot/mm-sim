import math
from dataclasses import replace

import numpy as np
import pytest

from mmsim import lvr
from mmsim import venues as vn

FAST = vn.Market(P0=100.0, sigma=0.5, lam=50.0, noise=20.0, size=0.5, hours=4.0)


def se(a):
    return np.std(a, ddof=1) / math.sqrt(len(a))


def test_ledgers_add_up_to_marked_pnl():
    s = vn.run(FAST, 0.3, 1e4, 0.004, vn.Quoter(half=0.003, gamma=0.01, limit=5), paths=50)
    for book in (s.lp, s.mm):
        assert book.inv + book.hedged == pytest.approx(book.total, abs=1e-8 * 1e4)
    assert s.mm.noise_fees.sum() == 0 and s.mm.arb_fees.sum() == 0
    assert np.all(s.mm.arb <= 0)


def test_arbitrage_profit_on_a_constant_product_pool_is_lvr_times_p_trade():
    m = vn.Market(P0=1.0, sigma=1.0, lam=50.0, noise=0.0, hours=10.0)
    g = 0.1
    s = vn.run(m, math.inf, 1.0, g, paths=400, seed=3)
    arb = -s.lp.arb - s.lp.arb_fees
    want = s.lvr * lvr.arb_rate(m.sigma, m.lam, g) / lvr.lvr_rate(m.sigma)
    assert abs(arb.mean() - want.mean()) < 4 * se(arb - want)
    fee = s.lp.arb_fees
    want = s.lvr * lvr.fee_rate(m.sigma, m.lam, g) / lvr.lvr_rate(m.sigma)
    assert abs(fee.mean() - want.mean()) < 4 * se(fee - want)


def test_noise_and_arbitrage_edges_sum_to_minus_lvr():
    s = vn.run(FAST, 0.5, 1e4, 0.003, paths=400, seed=6)
    edge = s.lp.noise + s.lp.arb
    assert s.lp.noise.mean() > 0
    assert abs(edge.mean() + s.lvr.mean()) < 4 * se(edge + s.lvr)


def test_quoter_loses_the_stale_quote_tail():
    h = 0.004
    s = vn.run(FAST, 0.3, 1e4, 0.05, vn.Quoter(half=h, limit=1e12), paths=400, seed=2)
    m = FAST
    drift = math.expm1(m.sigma**2 * m.hours / 2) / (m.sigma**2 * m.hours / 2)
    want = m.lam * m.hours * m.size * m.P0 * drift * lvr.stale_quote_loss(m.sigma, m.lam, h)
    assert abs(-s.mm.arb.mean() - want) < 4 * se(s.mm.arb)


def test_noise_goes_to_the_cheaper_venue():
    calm = replace(FAST, sigma=0.01)
    tight = vn.run(calm, math.inf, 1e6, 0.01, vn.Quoter(half=0.0005, limit=1e12), paths=20)
    assert tight.lp.volume.sum() < 0.1 * tight.mm.volume.sum()
    wide = vn.run(calm, math.inf, 1e6, 0.0005, vn.Quoter(half=0.05, limit=1e12), paths=20)
    assert wide.mm.volume.sum() == 0 and wide.lp.volume.sum() > 0


def test_inventory_limit_sends_flow_to_the_pool():
    s = vn.run(FAST, 0.3, 1e4, 0.01, vn.Quoter(half=0.0005, limit=1.0), paths=50, seed=1)
    assert s.lp.volume.sum() > 0
    assert s.mm.volume.sum() > 0


def test_narrow_range_runs_out_of_liquidity():
    s = vn.run(FAST, 0.02, 1e4, 0.003, paths=50, seed=4)
    assert s.dropped.sum() > 0
    assert vn.run(FAST, math.inf, 1e4, 0.003, paths=50, seed=4).dropped.sum() == 0
