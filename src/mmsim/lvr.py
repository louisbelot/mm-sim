from __future__ import annotations

import math

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


def arb_profit(z, gamma: float) -> np.ndarray:
    up = math.exp(gamma / 2) * (np.cosh((z - gamma) / 2) - 1)
    down = math.exp(-gamma / 2) * (np.cosh((z + gamma) / 2) - 1)
    return np.where(z > gamma, up, np.where(z < -gamma, down, 0.0))


def fee_income(z, gamma: float) -> np.ndarray:
    s = math.sinh(gamma / 2)
    up = s * (1 - np.exp(-(z - gamma) / 2))
    down = s * (np.exp(-(z + gamma) / 2) - 1)
    return np.where(z > gamma, up, np.where(z < -gamma, down, 0.0))
