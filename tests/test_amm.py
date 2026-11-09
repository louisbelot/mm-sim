import math

import numpy as np
import pytest

from mmsim import lvr
from mmsim.amm import Pool


def test_liquidity_matches_uniswap_v3_book():
    P, pa, pb = 5000.0, 4545.0, 5500.0
    L0 = 1 / (1 / math.sqrt(P) - 1 / math.sqrt(pb))
    L1 = 5000 / (math.sqrt(P) - math.sqrt(pa))
    assert L0 == pytest.approx(1519.437308014769733632, rel=1e-12)
    assert L1 == pytest.approx(1517.882343751509868544, rel=1e-12)
    x, y = Pool(min(L0, L1), math.sqrt(P), pa, pb).reserves()
    assert x == pytest.approx(0.998976618347425408, rel=1e-12)
    assert y == pytest.approx(5000, rel=1e-12)


@pytest.mark.parametrize("P", [50.0, 81.0, 100.0, 140.0, 300.0])
def test_virtual_reserves_keep_constant_product(P):
    pool = Pool(7.0, math.sqrt(P), 80.0, 150.0)
    x, y = pool.reserves()
    assert x >= 0 and y >= 0
    if 80 < P < 150:
        assert (x + pool.L / pool.sb) * (y + pool.L * pool.sa) == pytest.approx(pool.L**2)


def test_out_of_range_is_one_asset():
    pool = Pool(3.0, 1.0, 4.0, 9.0)
    assert pool.reserves()[1] == 0
    assert pool.reserves(np.sqrt(20.0))[0] == 0
    full = Pool(3.0, 2.0)
    x, y = full.reserves()
    assert x * y == pytest.approx(9.0)


def test_around_puts_capital_in_the_pool():
    for width in (0.05, 0.4, math.inf):
        pool = Pool.around(2000.0, width, 1e6, 0.003, n=3)
        assert pool.value(2000.0) == pytest.approx(np.full(3, 1e6))


def test_rebalanced_value_has_slope_x_and_lvr_is_its_curvature():
    pool = Pool(5.0, 10.0, 60.0, 160.0)
    sigma = 0.7
    for P in (70.0, 100.0, 150.0):
        h = 1e-3
        slope = (pool.value_at(P + h) - pool.value_at(P - h)) / (2 * h)
        assert slope == pytest.approx(pool.reserves(math.sqrt(P))[0], rel=1e-6)
        dx = (pool.reserves(math.sqrt(P + h))[0] - pool.reserves(math.sqrt(P - h))[0]) / (2 * h)
        assert pool.lvr_rate(P, sigma) == pytest.approx(-(sigma**2) * P**2 / 2 * dx, rel=1e-6)
    assert pool.lvr_rate(50.0, sigma) == 0 and pool.lvr_rate(170.0, sigma) == 0


def test_round_trip_costs_exactly_the_fees():
    pool = Pool(100.0, 10.0, gamma=0.01)
    s0 = pool.sp.copy()
    target, ok = pool.buy_target(2.0)
    dx, dy, fee_in = pool.swap_to(target)
    assert ok and dx == pytest.approx(-2.0)
    target, ok = pool.sell_target(2.0)
    dx2, dy2, fee_out = pool.swap_to(target)
    assert ok and dx2 == pytest.approx(2.0)
    assert pool.sp == pytest.approx(s0)
    assert dy + dy2 == pytest.approx(0, abs=1e-9)
    assert fee_in == pytest.approx(math.expm1(0.01) * dy)
    assert fee_out == pytest.approx(-math.expm1(-0.01) * dy)
    assert pool.fees == pytest.approx(fee_in + fee_out)


def test_cannot_trade_past_the_range():
    pool = Pool(10.0, 3.0, 4.0, 16.0)
    x, _ = pool.reserves()
    assert not pool.buy_target(x + 1e-9)[1]
    assert pool.buy_target(x - 1e-9)[1]
    cap = pool.L * (1 / pool.sa - 1 / pool.sp)
    assert not pool.sell_target(cap + 1e-9)[1]
    assert pool.sell_target(cap - 1e-9)[1]


def test_arb_stops_at_the_edge_of_the_no_trade_band():
    g = 0.02
    for P in (1.1, 0.9):
        pool = Pool(1.0, 1.0, gamma=g)
        pool.swap_to(pool.arb_target(P))
        assert math.log(P / pool.price) == pytest.approx(g if P > 1 else -g)
    pool = Pool(1.0, 1.0, gamma=g)
    for P in (math.exp(0.019), math.exp(-0.019)):
        assert pool.arb_target(P) == pool.sp


@pytest.mark.parametrize("z", [-0.6, -0.11, 0.05, 0.11, 0.6])
def test_cpmm_arb_matches_normalized_profit(z):
    g = 0.1
    P = math.exp(z)
    pool = Pool(2.5, 1.0, gamma=g)
    dx, dy, fee = pool.swap_to(pool.arb_target(P))
    V = pool.value_at(P)
    assert (-(P * dx + dy) - fee) / V == pytest.approx(lvr.arb_profit(z, g), abs=1e-14)
    assert fee / V == pytest.approx(lvr.fee_income(z, g), abs=1e-14)


@pytest.mark.parametrize("z", [-0.6, -0.2, 0.17, 0.6])
def test_hedged_arb_matches_normalized_profit(z):
    g, h = 0.1, 0.05
    P = math.exp(z)
    pool = Pool(2.5, 1.0, gamma=g)
    dx, dy, fee = pool.swap_to(pool.arb_target(P, h))
    hedge_px = P * math.exp(-h if dx < 0 else h)
    arb = -(hedge_px * dx + dy) - fee
    quoter = (hedge_px - P) * dx
    V = pool.value_at(P)
    assert arb / V == pytest.approx(lvr.arb_profit(z, g, h), abs=1e-14)
    assert fee / V == pytest.approx(lvr.fee_income(z, g, h), abs=1e-14)
    assert quoter / V == pytest.approx(lvr.hedge_income(z, g, h), abs=1e-14)
    assert arb + quoter + (P * dx + dy) + fee == pytest.approx(0, abs=1e-14)


def test_hedge_cost_widens_the_no_trade_band():
    g, h = 0.02, 0.01
    pool = Pool(1.0, 1.0, gamma=g)
    for P in (math.exp(0.029), math.exp(-0.029)):
        assert pool.arb_target(P, h) == pool.sp
    for P in (math.exp(0.031), math.exp(-0.031)):
        assert pool.arb_target(P, h) != pool.sp


def test_per_path_fee_matches_separate_pools():
    g = np.array([0.0005, 0.003, 0.02])
    sp = np.sqrt([100.0, 101.0, 99.0])
    pool = Pool(50.0, sp, 80.0, 150.0, g)
    P = np.array([100.4, 98.7, 103.0])
    dx, dy, fee = pool.swap_to(pool.arb_target(P))
    for i in range(3):
        one = Pool(50.0, sp[i : i + 1], 80.0, 150.0, float(g[i]))
        want = one.swap_to(one.arb_target(P[i : i + 1]))
        assert (dx[i], dy[i], fee[i]) == pytest.approx([w[0] for w in want], rel=1e-14)
    assert pool.sp[2] == pytest.approx(math.sqrt(103.0) * math.exp(-0.01))
