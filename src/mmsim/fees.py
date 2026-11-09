from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Static:
    gamma: float

    def __call__(self, sigma) -> np.ndarray:
        return np.full(np.shape(sigma), self.gamma)


@dataclass(frozen=True)
class Threshold:
    low: float
    high: float
    cut: float

    def __call__(self, sigma) -> np.ndarray:
        return np.where(np.asarray(sigma) > self.cut, self.high, self.low)


@dataclass(frozen=True)
class Power:
    base: float
    k: float
    p: float = 1.0
    cap: float = 0.1
    floor: float = 0.0

    def __call__(self, sigma) -> np.ndarray:
        return np.clip(self.base + self.k * np.asarray(sigma) ** self.p, self.floor, self.cap)


@dataclass(frozen=True)
class Regimes:
    spike: float = 3.0
    up: float = 1 / 20
    down: float = 1 / 4

    @property
    def share(self) -> float:
        return self.up / (self.up + self.down)

    def rms(self, sigma: float) -> float:
        return sigma * math.sqrt(1 + self.share * (self.spike**2 - 1))


class Ewma:
    def __init__(self, sigma0, half_life: float):
        self.H = half_life / math.log(2)
        self.v = np.asarray(sigma0, dtype=float) ** 2 * self.H

    def update(self, r, tau) -> None:
        self.v = self.v * np.exp(-tau / self.H) + np.square(r)

    @property
    def sigma(self) -> np.ndarray:
        return np.sqrt(self.v / self.H)
