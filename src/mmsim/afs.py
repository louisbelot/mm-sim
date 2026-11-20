from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np
from scipy.optimize import brentq

Resilience = Literal["volume", "price"]


class Shape(Protocol):
    def f(self, x: float) -> float: ...
    def F(self, x: float) -> float: ...
    def inv(self, y: float) -> float: ...
    def G(self, x: float) -> float: ...


@dataclass(frozen=True)
class Block:
    q: float

    def f(self, x: float) -> float:
        return self.q

    def F(self, x: float) -> float:
        return self.q * x

    def inv(self, y: float) -> float:
        return y / self.q

    def G(self, x: float) -> float:
        return 0.5 * self.q * x * x


@dataclass(frozen=True)
class Power:
    q: float
    alpha: float

    def __post_init__(self) -> None:
        if self.alpha >= 1.0:
            raise ValueError("alpha >= 1 leaves the book finite")

    def f(self, x: float) -> float:
        return self.q * (1.0 + x) ** -self.alpha

    def F(self, x: float) -> float:
        b = 1.0 - self.alpha
        return self.q * ((1.0 + x) ** b - 1.0) / b

    def inv(self, y: float) -> float:
        b = 1.0 - self.alpha
        return (1.0 + b * y / self.q) ** (1.0 / b) - 1.0

    def G(self, x: float) -> float:
        b = 1.0 - self.alpha
        u = 1.0 + x
        return self.q * ((u ** (b + 1.0) - 1.0) / (b + 1.0) - (u**b - 1.0) / b)


@dataclass(frozen=True)
class Profile:
    depth: tuple[float, ...]
    _vol: np.ndarray = field(init=False, repr=False, compare=False)
    _area: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        d = np.asarray(self.depth, dtype=float)
        if d.size == 0 or np.any(d <= 0.0):
            raise ValueError("depth must be positive")
        k = np.arange(d.size)
        object.__setattr__(self, "_vol", np.concatenate([[0.0], np.cumsum(d)]))
        object.__setattr__(self, "_area", np.concatenate([[0.0], np.cumsum(d * (k + 0.5))]))

    def _at(self, x: float) -> int:
        return min(int(x), len(self.depth) - 1)

    def f(self, x: float) -> float:
        return self.depth[self._at(x)]

    def F(self, x: float) -> float:
        k = self._at(x)
        return float(self._vol[k] + self.depth[k] * (x - k))

    def inv(self, y: float) -> float:
        k = min(int(np.searchsorted(self._vol, y, side="right")) - 1, len(self.depth) - 1)
        return k + (y - self._vol[k]) / self.depth[k]

    def G(self, x: float) -> float:
        k = self._at(x)
        return float(self._area[k] + 0.5 * self.depth[k] * (x * x - k * k))


def cost(shape: Shape, xi: Sequence[float], a: float, model: Resilience) -> float:
    d, total = 0.0, 0.0
    for x in xi:
        up = shape.inv(shape.F(d) + x)
        total += shape.G(up) - shape.G(d)
        d = shape.inv(a * shape.F(up)) if model == "volume" else a * up
    return total


def _trickle(shape: Shape, x0: float, a: float, model: Resilience) -> float:
    if model == "volume":
        return (1.0 - a) * x0
    return x0 - shape.F(a * shape.inv(x0))


def _target(shape: Shape, x0: float, a: float, model: Resilience) -> float:
    if model == "volume":
        return (shape.inv(x0) - a * shape.inv(a * x0)) / (1.0 - a)
    d = shape.inv(x0)
    fd, fa = shape.f(d), shape.f(a * d)
    return d * (fd - a * a * fa) / (fd - a * fa)


def optimal(shape: Shape, X: float, N: int, a: float, model: Resilience) -> np.ndarray:
    if N < 1 or not 0.0 <= a < 1.0 or X <= 0.0:
        raise ValueError("need N >= 1, 0 <= a < 1 and X > 0")

    def trickle(x0):
        return _trickle(shape, x0, a, model)

    def gap(x0):
        return shape.inv(X - N * trickle(x0)) - _target(shape, x0, a, model)

    hi = brentq(lambda x0: x0 + (N - 1) * trickle(x0) - X, 0.0, X, xtol=1e-14 * X)
    x0 = brentq(gap, 1e-12 * hi, hi, xtol=1e-14 * X)
    mid = trickle(x0)
    return np.array([x0, *[mid] * (N - 1), X - x0 - (N - 1) * mid])


def continuous(shape: Shape, X: float, rT: float) -> tuple[float, float, float]:
    def gap(x0):
        d = shape.inv(x0)
        return shape.inv(X - rT * x0) - d - x0 / shape.f(d)

    x0 = brentq(gap, 1e-12 * X, X / (1.0 + rT), xtol=1e-14 * X)
    return x0, rT * x0, X - x0 * (1.0 + rT)
