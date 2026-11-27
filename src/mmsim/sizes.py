from __future__ import annotations

import math
import random
from bisect import bisect_right
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

ITCH_ALPHA = 2.7
ITCH_CAP = 20


@dataclass(frozen=True)
class Sizes:
    pmf: tuple[float, ...]
    _cum: tuple[float, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        p = np.asarray(self.pmf, dtype=float)
        if p.ndim != 1 or not len(p) or np.any(p < 0) or abs(p.sum() - 1) > 1e-9:
            raise ValueError("pmf has to be a probability vector over sizes 1, 2, ...")
        object.__setattr__(self, "_cum", tuple(np.cumsum(p)))

    @classmethod
    def unit(cls) -> Sizes:
        return cls((1.0,))

    @classmethod
    def zeta(cls, alpha: float, cap: int) -> Sizes:
        w = np.arange(1, cap + 1, dtype=float) ** -alpha
        return cls(tuple(w / w.sum()))

    @classmethod
    def geometric(cls, q: float, cap: int) -> Sizes:
        w = q ** np.arange(cap, dtype=float)
        return cls(tuple(w / w.sum()))

    @classmethod
    def itch(cls) -> Sizes:
        return cls.zeta(ITCH_ALPHA, ITCH_CAP)

    @property
    def cap(self) -> int:
        return len(self.pmf)

    @property
    def mean(self) -> float:
        return float(np.arange(1, self.cap + 1) @ np.asarray(self.pmf))

    def draw(self, rng: random.Random) -> int:
        return 1 + bisect_right(self._cum, rng.random() * self._cum[-1], 0, self.cap - 1)

    def logpmf(self, k: np.ndarray) -> np.ndarray:
        return np.log(np.asarray(self.pmf))[np.asarray(k) - 1]


def units(shares: np.ndarray, unit: float, cap: int) -> np.ndarray:
    return np.clip(np.round(np.asarray(shares) / unit), 1, cap).astype(int)


def _match(stat: np.ndarray, target: float, family, lo: float, hi: float) -> float:
    def gap(x: float) -> float:
        return float(np.asarray(family(x).pmf) @ stat) - target

    return brentq(gap, lo, hi, xtol=1e-12)


def fit_zeta(k: np.ndarray, cap: int) -> Sizes:
    stat = np.log(np.arange(1, cap + 1))
    target = float(np.mean(np.log(k)))
    alpha = _match(stat, target, lambda a: Sizes.zeta(a, cap), -5.0, 30.0)
    return Sizes.zeta(alpha, cap)


def fit_geometric(k: np.ndarray, cap: int) -> Sizes:
    stat = np.arange(cap, dtype=float)
    target = float(np.mean(k)) - 1
    q = _match(stat, target, lambda q: Sizes.geometric(q, cap), 1e-12, 1 - 1e-9)
    return Sizes.geometric(q, cap)


def zeta_alpha(s: Sizes) -> float:
    p = np.log(np.asarray(s.pmf))
    return float(p[0] - p[1]) / math.log(2)


def vuong(a: Sizes, b: Sizes, k: np.ndarray) -> tuple[float, float]:
    d = a.logpmf(k) - b.logpmf(k)
    z = float(d.sum() / (math.sqrt(len(d)) * d.std()))
    return float(d.sum()), z
