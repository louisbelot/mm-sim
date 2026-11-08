from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def eta(sigma: float, lam: float, gamma: float) -> float:
    return math.sqrt(2 * lam) * gamma / sigma


def p_trade(sigma: float, lam: float, gamma: float) -> float:
    return 1 / (1 + eta(sigma, lam, gamma))


def lvr_rate(sigma: float) -> float:
    return sigma**2 / 8


def arb_rate(sigma: float, lam: float, gamma: float) -> float:
    if lam <= sigma**2 / 8:
        return math.inf
    return (
        lvr_rate(sigma)
        * p_trade(sigma, lam, gamma)
        * math.cosh(gamma / 2)
        / (1 - sigma**2 / (8 * lam))
    )


def fee_rate(sigma: float, lam: float, gamma: float) -> float:
    zeta = math.sqrt(2 * lam) / sigma
    if zeta <= 0.5:
        return math.inf
    return lam * p_trade(sigma, lam, gamma) * math.sinh(gamma / 2) * 2 * zeta / (4 * zeta**2 - 1)


def stale_quote_loss(sigma: float, lam: float, half: float) -> float:
    b = sigma / math.sqrt(2 * lam)
    if b >= 1:
        return math.inf
    c, d = math.log1p(half), math.log1p(-half)
    return b / 2 * (math.exp(c * (1 - 1 / b)) / (1 - b) + math.exp(d * (1 + 1 / b)) / (1 + b))


def arb_profit(z, gamma: float) -> np.ndarray:
    up = math.exp(gamma / 2) * (np.cosh((z - gamma) / 2) - 1)
    down = math.exp(-gamma / 2) * (np.cosh((z + gamma) / 2) - 1)
    return np.where(z > gamma, up, np.where(z < -gamma, down, 0.0))


def fee_income(z, gamma: float) -> np.ndarray:
    s = math.sinh(gamma / 2)
    up = s * (1 - np.exp(-(z - gamma) / 2))
    down = s * (np.exp(-(z + gamma) / 2) - 1)
    return np.where(z > gamma, up, np.where(z < -gamma, down, 0.0))


@dataclass(frozen=True)
class Chain:
    arb: float
    fee: float
    trade: float
    arb_se: float
    fee_se: float


def mispricing(
    sigma: float,
    lam: float,
    gamma: float,
    *,
    chains: int = 2000,
    blocks: int = 2000,
    burn: int = 200,
    fixed: bool = False,
    seed: int = 0,
) -> Chain:
    rng = np.random.default_rng(seed)
    z = np.zeros(chains)
    arb = np.zeros(chains)
    fee = np.zeros(chains)
    time = np.zeros(chains)
    trades = 0
    for i in range(burn + blocks):
        tau = np.full(chains, 1 / lam) if fixed else rng.exponential(1 / lam, chains)
        z += sigma * np.sqrt(tau) * rng.standard_normal(chains)
        if i >= burn:
            arb += arb_profit(z, gamma)
            fee += fee_income(z, gamma)
            time += tau
            trades += np.count_nonzero(np.abs(z) > gamma)
        np.clip(z, -gamma, gamma, out=z)
    T = time.sum()
    return Chain(
        arb.sum() / T,
        fee.sum() / T,
        trades / (chains * blocks),
        np.std(arb / time, ddof=1) / math.sqrt(chains),
        np.std(fee / time, ddof=1) / math.sqrt(chains),
    )
