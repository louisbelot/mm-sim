from __future__ import annotations

import math

import numpy as np


class Pool:
    def __init__(self, L: float, sp, pa: float = 0.0, pb: float = math.inf, gamma: float = 0.003):
        if not 0 <= pa < pb:
            raise ValueError("need 0 <= pa < pb")
        self.L = L
        self.sa = math.sqrt(pa)
        self.sb = math.sqrt(pb)
        self.gamma = gamma
        self.sp = np.clip(np.array(sp, dtype=float), self.sa, self.sb)
        self.fees = np.zeros_like(self.sp)

    @classmethod
    def around(cls, P: float, width: float, capital: float, gamma: float, n: int = 1) -> Pool:
        if math.isinf(width):
            pa, pb = 0.0, math.inf
        else:
            pa, pb = P * math.exp(-width), P * math.exp(width)
        unit = 2 * math.sqrt(P) - math.sqrt(pa) - P / math.sqrt(pb)
        return cls(capital / unit, np.full(n, math.sqrt(P)), pa, pb, gamma)

    @property
    def pa(self) -> float:
        return self.sa**2

    @property
    def pb(self) -> float:
        return self.sb**2

    @property
    def price(self) -> np.ndarray:
        return self.sp**2

    def reserves(self, sp=None) -> tuple[np.ndarray, np.ndarray]:
        s = self.sp if sp is None else np.clip(sp, self.sa, self.sb)
        return self.L * (1 / s - 1 / self.sb), self.L * (s - self.sa)

    def value(self, P) -> np.ndarray:
        x, y = self.reserves()
        return P * x + y

    def value_at(self, P) -> np.ndarray:
        x, y = self.reserves(np.sqrt(P))
        return P * x + y

    def lvr_rate(self, P, sigma: float) -> np.ndarray:
        inside = (P > self.pa) & (P < self.pb)
        return np.where(inside, sigma**2 / 4 * self.L * np.sqrt(P), 0.0)

    def flows(self, s_new) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        s_new = np.clip(s_new, self.sa, self.sb)
        dx = self.L * (1 / s_new - 1 / self.sp)
        dy = self.L * (s_new - self.sp)
        fee = np.where(dy > 0, math.expm1(self.gamma), math.expm1(-self.gamma)) * dy
        return dx, dy, fee

    def swap_to(self, s_new) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dx, dy, fee = self.flows(s_new)
        self.sp = np.clip(s_new, self.sa, self.sb)
        self.fees = self.fees + fee
        return dx, dy, fee

    def arb_target(self, P, h: float = 0.0) -> np.ndarray:
        up, down = P * math.exp(-h), P * math.exp(h)
        p = self.price
        g = self.gamma
        return np.where(
            up > p * math.exp(g),
            np.sqrt(up) * math.exp(-g / 2),
            np.where(down < p * math.exp(-g), np.sqrt(down) * math.exp(g / 2), self.sp),
        )

    def buy_target(self, qty) -> tuple[np.ndarray, np.ndarray]:
        inv = 1 / self.sp - qty / self.L
        ok = inv > 1 / self.sb
        return np.where(ok, 1 / np.where(ok, inv, 1.0), self.sp), ok

    def sell_target(self, qty) -> tuple[np.ndarray, np.ndarray]:
        s_new = 1 / (1 / self.sp + qty / self.L)
        ok = s_new > self.sa
        return np.where(ok, s_new, self.sp), ok
