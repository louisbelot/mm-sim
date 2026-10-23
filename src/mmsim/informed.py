from __future__ import annotations

import math
import random

from .book import OrderType, Side
from .sim import Agent, Context


class Fundamental:
    def __init__(self, sigma: float, seed: int = 0) -> None:
        self.sigma = sigma
        self.rng = random.Random(seed)
        self.t: float | None = None
        self.v = math.nan
        self.path: list[tuple[float, float]] = []

    def start(self, t: float, v: float) -> None:
        self.t, self.v = t, v
        self.path = [(t, v)]

    def at(self, t: float) -> float:
        if t > self.t:
            self.v += self.sigma * math.sqrt(t - self.t) * self.rng.gauss(0.0, 1.0)
            self.t = t
            self.path.append((t, self.v))
        return self.v


class InformedTrader(Agent):
    def __init__(
        self, value: Fundamental, rate: float, *, edge: float = 0.0, size: int = 1, seed: int = 0
    ) -> None:
        self.value = value
        self.rate = rate
        self.edge = edge
        self.size = size
        self.rng = random.Random(seed)

    def on_start(self, ctx: Context) -> None:
        self._begin(ctx)
        ctx.set_timer(self.rng.expovariate(self.rate))

    def on_timer(self, ctx: Context) -> None:
        if not self._begin(ctx):
            ctx.set_timer(self.rng.expovariate(self.rate))
            return
        v = self.value.at(ctx.t)
        bid, ask = ctx.book.best_bid(), ctx.book.best_ask()
        if ask is not None and v > ask + self.edge:
            ctx.submit(Side.BUY, 0, self.size, kind=OrderType.MARKET)
        elif bid is not None and v < bid - self.edge:
            ctx.submit(Side.SELL, 0, self.size, kind=OrderType.MARKET)
        ctx.set_timer(self.rng.expovariate(self.rate))

    def _begin(self, ctx: Context) -> bool:
        if self.value.t is None:
            mid = ctx.book.mid()
            if mid is None:
                return False
            self.value.start(ctx.t, mid)
        return True
