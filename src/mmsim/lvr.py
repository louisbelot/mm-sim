from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import zeta

ZETA_HALF = float(zeta(0.5))


def eta(sigma: float, lam: float, gamma: float) -> float:
    return math.sqrt(2 * lam) * gamma / sigma


def p_trade(sigma: float, lam: float, gamma: float) -> float:
    return 1 / (1 + eta(sigma, lam, gamma))


def lvr_rate(sigma: float) -> float:
    return sigma**2 / 8


def arb_rate(sigma: float, lam: float, gamma: float, h: float = 0.0) -> float:
    if lam <= sigma**2 / 8:
        return math.inf
    return (
        lvr_rate(sigma)
        * p_trade(sigma, lam, gamma + h)
        * math.cosh((gamma - h) / 2)
        / (1 - sigma**2 / (8 * lam))
    )


def _tails(sigma: float, lam: float, up: float, down: float) -> float:
    z = math.sqrt(2 * lam) / sigma
    if z <= 0.5:
        return math.inf
    return (up / (2 * z - 1) + down / (2 * z + 1)) / 2


def fee_rate(sigma: float, lam: float, gamma: float, h: float = 0.0) -> float:
    t = _tails(sigma, lam, math.exp(h / 2), math.exp(-h / 2))
    return lam * p_trade(sigma, lam, gamma + h) * math.sinh(gamma / 2) * t


def hedge_rate(sigma: float, lam: float, gamma: float, h: float) -> float:
    t = _tails(sigma, lam, math.exp(gamma / 2), math.exp(-gamma / 2))
    return lam * p_trade(sigma, lam, gamma + h) * math.sinh(h / 2) * t


def p_trade_fixed(sigma: float, lam: float, gamma: float) -> float:
    return 1 / (eta(sigma, lam, gamma) + abs(ZETA_HALF) / math.sqrt(math.pi))


def arb_rate_fixed(sigma: float, lam: float, gamma: float) -> float:
    return lvr_rate(sigma) / (1 + math.sqrt(math.pi) * eta(sigma, lam, gamma) / abs(ZETA_HALF))


def stale_quote_loss(sigma: float, lam: float, half: float) -> float:
    b = sigma / math.sqrt(2 * lam)
    if b >= 1:
        return math.inf
    c, d = math.log1p(half), math.log1p(-half)
    return b / 2 * (math.exp(c * (1 - 1 / b)) / (1 - b) + math.exp(d * (1 + 1 / b)) / (1 + b))


def _sides(z, g: float, up, down) -> np.ndarray:
    return np.where(z > g, up, np.where(z < -g, down, 0.0))


def arb_profit(z, gamma: float, h: float = 0.0) -> np.ndarray:
    g = gamma + h
    up = math.exp((gamma - h) / 2) * (np.cosh((z - g) / 2) - 1)
    down = math.exp((h - gamma) / 2) * (np.cosh((z + g) / 2) - 1)
    return _sides(z, g, up, down)


def fee_income(z, gamma: float, h: float = 0.0) -> np.ndarray:
    g, s = gamma + h, math.sinh(gamma / 2)
    up = math.exp(-h / 2) * s * (1 - np.exp(-(z - g) / 2))
    down = math.exp(h / 2) * s * (np.exp(-(z + g) / 2) - 1)
    return _sides(z, g, up, down)


def hedge_income(z, gamma: float, h: float) -> np.ndarray:
    g, s = gamma + h, math.sinh(h / 2)
    up = math.exp(gamma / 2) * s * (np.exp((z - g) / 2) - 1)
    down = math.exp(-gamma / 2) * s * (1 - np.exp((z + g) / 2))
    return _sides(z, g, up, down)


@dataclass(frozen=True)
class Chain:
    arb: float
    fee: float
    trade: float
    arb_se: float
    fee_se: float
    hedge: float = 0.0
    hedge_se: float = 0.0


def mispricing(
    sigma: float,
    lam: float,
    gamma: float,
    *,
    chains: int = 2000,
    blocks: int = 2000,
    burn: int = 200,
    fixed: bool = False,
    h: float = 0.0,
    seed: int = 0,
) -> Chain:
    rng = np.random.default_rng(seed)
    z = np.zeros(chains)
    arb = np.zeros(chains)
    fee = np.zeros(chains)
    hedge = np.zeros(chains)
    time = np.zeros(chains)
    g = gamma + h
    trades = 0
    for i in range(burn + blocks):
        tau = np.full(chains, 1 / lam) if fixed else rng.exponential(1 / lam, chains)
        z += sigma * np.sqrt(tau) * rng.standard_normal(chains)
        if i >= burn:
            arb += arb_profit(z, gamma, h)
            fee += fee_income(z, gamma, h)
            hedge += hedge_income(z, gamma, h)
            time += tau
            trades += np.count_nonzero(np.abs(z) > g)
        np.clip(z, -g, g, out=z)
    T = time.sum()
    return Chain(
        arb.sum() / T,
        fee.sum() / T,
        trades / (chains * blocks),
        np.std(arb / time, ddof=1) / math.sqrt(chains),
        np.std(fee / time, ddof=1) / math.sqrt(chains),
        hedge.sum() / T,
        np.std(hedge / time, ddof=1) / math.sqrt(chains),
    )
