from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from mmsim.amm import Pool
from mmsim.avellaneda import reservation_price
from mmsim.fees import Ewma, Regimes, Static


@dataclass(frozen=True)
class Market:
    P0: float = 2000.0
    sigma: float = 0.6 / math.sqrt(8760)
    lam: float = 300.0
    noise: float = 60.0
    size: float = 1.0
    hours: float = 24.0


@dataclass(frozen=True)
class Quoter:
    half: float = 0.0005
    gamma: float = 0.0
    tau: float = 1.0
    limit: float = 20.0
    widen: bool = False


@dataclass
class Ledger:
    noise: np.ndarray
    arb: np.ndarray
    noise_fees: np.ndarray
    arb_fees: np.ndarray
    inv: np.ndarray
    volume: np.ndarray
    arb_volume: np.ndarray
    total: np.ndarray

    @property
    def hedged(self) -> np.ndarray:
        return self.noise + self.arb + self.noise_fees + self.arb_fees

    @classmethod
    def empty(cls, n: int) -> Ledger:
        return cls(*(np.zeros(n) for _ in range(8)))


@dataclass(frozen=True)
class Session:
    lp: Ledger
    mm: Ledger | None
    lvr: np.ndarray
    dropped: np.ndarray
    fee: np.ndarray
    spike: np.ndarray
    lvr_spike: np.ndarray


def run(
    market: Market,
    width: float,
    capital: float,
    gamma: float | Callable[[np.ndarray], np.ndarray],
    quoter: Quoter | None = None,
    *,
    regimes: Regimes | None = None,
    source: str = "market",
    half_life: float = 1.0,
    paths: int = 200,
    seed: int = 0,
) -> Session:
    if source not in ("true", "market", "pool"):
        raise ValueError(f"unknown source {source!r}")
    rng = np.random.default_rng(seed)
    m = market
    rule = gamma if callable(gamma) else Static(gamma)
    sig = np.full(paths, m.sigma)
    est = Ewma(sig, half_life)
    pool = Pool.around(m.P0, width, capital, 0.0, paths)
    pool.gamma = rule(sig)
    v0 = pool.value(m.P0)
    lp = Ledger.empty(paths)
    mm = Ledger.empty(paths) if quoter else None
    q = np.zeros(paths)
    cash = np.zeros(paths)
    bid = np.full(paths, -np.inf)
    ask = np.full(paths, np.inf)
    P = np.full(paths, m.P0)
    t = np.zeros(paths)
    lvr = np.zeros(paths)
    dropped = np.zeros(paths)
    fee_time = np.zeros(paths)
    spike = np.zeros(paths)
    lvr_spike = np.zeros(paths)
    hot = np.zeros(paths, dtype=bool)
    s_seen = pool.sp
    rate = m.lam + m.noise
    alive = np.ones(paths, dtype=bool)

    def swap(mask, target, edge, fees, vol):
        dx, dy, fee = pool.swap_to(np.where(mask, target, pool.sp))
        edge += P * dx + dy
        fees += fee
        vol += np.abs(dx)

    def requote(mask):
        r = reservation_price(P, q, quoter.tau, quoter.gamma, sig * P)
        half = quoter.half * sig / m.sigma if quoter.widen else quoter.half
        bid[:] = np.where(mask, r * (1 - half), bid)
        ask[:] = np.where(mask, r * (1 + half), ask)

    def can_sell():
        return q - m.size >= -quoter.limit

    def can_buy():
        return q + m.size <= quoter.limit

    def fill(sold, bought, edge, vol):
        n = m.size * (bought.astype(float) - sold)
        px = np.where(sold, ask, bid)
        traded = sold | bought
        edge += np.where(traded, (P - px) * n, 0.0)
        vol += np.abs(n)
        cash[:] -= np.where(traded, px * n, 0.0)
        q[:] += n

    if quoter:
        requote(alive)

    while alive.any():
        tau = np.minimum(rng.exponential(1 / rate, paths), m.hours - t)
        tau = np.where(alive, tau, 0.0)
        x, _ = pool.reserves()
        accrued = pool.lvr_rate(P, sig) * tau
        lvr += accrued
        lvr_spike += np.where(hot, accrued, 0.0)
        spike += np.where(hot, tau, 0.0)
        fee_time += pool.gamma * tau
        dP = P * np.expm1(sig * np.sqrt(tau) * rng.standard_normal(paths))
        lp.inv += x * dP
        if mm:
            mm.inv += q * dP
        P_old = P
        P = P + dP
        t = t + tau
        alive &= t < m.hours
        if regimes:
            flip = np.where(hot, regimes.down, regimes.up)
            hot ^= rng.random(paths) < -np.expm1(-flip * tau)
            sig = np.where(hot, regimes.spike * m.sigma, m.sigma)
        if source == "market":
            est.update(np.log(P / P_old), tau)
        elif source == "pool":
            est.update(2 * np.log(pool.sp / s_seen), tau)
            s_seen = pool.sp
        seen = sig if source == "true" else est.sigma
        pool.gamma = np.where(alive, rule(seen), pool.gamma)
        block = alive & (rng.random(paths) < m.lam / rate)
        flow = alive & ~block

        swap(block, pool.arb_target(P), lp.arb, lp.arb_fees, lp.arb_volume)

        if mm:
            lift = block & (ask < P) & can_sell()
            hit = block & (bid > P) & can_buy()
            fill(lift, hit, mm.arb, mm.arb_volume)
            requote(block)

        buy = rng.random(paths) < 0.5
        sb, ok_b = pool.buy_target(m.size)
        ss, ok_s = pool.sell_target(m.size)
        target = np.where(buy, sb, ss)
        dx, dy, fee = pool.flows(target)
        cost = (dy + fee) / m.size
        pool_px = np.where(buy, np.where(ok_b, cost, np.inf), np.where(ok_s, -cost, -np.inf))
        if mm:
            mm_ask = np.where(can_sell(), ask, np.inf)
            mm_bid = np.where(can_buy(), bid, -np.inf)
            mm_px = np.where(buy, mm_ask, mm_bid)
            better = np.where(buy, mm_px <= pool_px, mm_px >= pool_px)
            to_mm = flow & better & np.isfinite(mm_px)
        else:
            to_mm = np.zeros(paths, dtype=bool)
        to_pool = flow & ~to_mm & np.isfinite(pool_px)
        dropped += flow & ~to_mm & ~to_pool

        swap(to_pool, target, lp.noise, lp.noise_fees, lp.volume)
        if mm:
            fill(to_mm & buy, to_mm & ~buy, mm.noise, mm.volume)

    lp.total[:] = pool.value(P) + pool.fees - v0
    if mm:
        mm.total[:] = cash + q * P
    fee = fee_time / m.hours
    return Session(lp, mm, lvr, dropped, fee, spike / m.hours, lvr_spike)
