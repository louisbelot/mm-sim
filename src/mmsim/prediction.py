from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from mmsim.glosten import binary_quotes


class LMSR:
    def __init__(self, b: float, n: int, p0: float = 0.5):
        self.b = b
        self.q = np.full(n, b * math.log(p0 / (1 - p0)))

    @property
    def price(self) -> np.ndarray:
        return expit(self.q / self.b)

    def cost(self, q) -> np.ndarray:
        return self.b * np.logaddexp(q / self.b, 0.0)

    def buy(self, yes: np.ndarray) -> np.ndarray:
        q = self.q + np.where(yes, 1.0, -1.0)
        paid = self.cost(q) - self.cost(self.q) + np.where(yes, 0.0, 1.0)
        self.q = q
        return paid


class FPMM:
    def __init__(self, F: float, n: int, p0: float = 0.5):
        self.F = F
        self.y = np.full(n, F * math.sqrt((1 - p0) / p0))
        self.n = np.full(n, F * math.sqrt(p0 / (1 - p0)))

    @property
    def price(self) -> np.ndarray:
        return self.n / (self.y + self.n)

    def buy(self, yes: np.ndarray) -> np.ndarray:
        out, other = np.where(yes, self.y, self.n), np.where(yes, self.n, self.y)
        s = out + other - 1
        c = (np.sqrt(s * s + 4 * other) - s) / 2
        self.y, self.n = (
            np.where(yes, out + c - 1, other + c),
            np.where(yes, other + c, out + c - 1),
        )
        return c


class Quoter:
    def __init__(self, alpha: float, n: int, p0: float = 0.5):
        self.alpha = alpha
        self.mu = np.full(n, p0)

    @property
    def price(self) -> np.ndarray:
        return self.mu

    def buy(self, yes: np.ndarray) -> np.ndarray:
        bid, ask = binary_quotes(self.mu, self.alpha, 0.0, 1.0)
        self.mu = np.where(yes, ask, bid)
        return np.where(yes, ask, 1 - bid)


@dataclass(frozen=True)
class Markets:
    error: np.ndarray
    maker: np.ndarray
    informed: np.ndarray
    noise: np.ndarray
    price: np.ndarray
    outcome: np.ndarray


def run(
    make, alpha: float, trades: int, markets: int, *, p0: float = 0.5, seed: int = 0
) -> Markets:
    rng = np.random.default_rng(seed)
    mm = make(markets)
    omega = rng.random(markets) < p0
    error = np.empty(trades + 1)
    error[0] = np.abs(mm.price - omega).mean()
    pnl = {"maker": np.zeros(markets), "informed": np.zeros(markets), "noise": np.zeros(markets)}
    for t in range(trades):
        informed = rng.random(markets) < alpha
        yes = np.where(informed, omega, rng.random(markets) < 0.5)
        paid = mm.buy(yes)
        gain = (yes == omega) - paid
        pnl["maker"] -= gain
        pnl["informed"] += np.where(informed, gain, 0.0)
        pnl["noise"] += np.where(informed, 0.0, gain)
        error[t + 1] = np.abs(mm.price - omega).mean()
    return Markets(error, pnl["maker"], pnl["informed"], pnl["noise"], mm.price, omega)
