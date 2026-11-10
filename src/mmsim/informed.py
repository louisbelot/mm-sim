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
        self._trade(ctx)
        ctx.set_timer(self.rng.expovariate(self.rate))

    def _trade(self, ctx: Context) -> None:
        v = self.value.at(ctx.t)
        bid, ask = ctx.book.best_bid(), ctx.book.best_ask()
        if ask is not None and v > ask + self.edge:
            ctx.submit(Side.BUY, 0, self.size, kind=OrderType.MARKET)
        elif bid is not None and v < bid - self.edge:
            ctx.submit(Side.SELL, 0, self.size, kind=OrderType.MARKET)

    def _begin(self, ctx: Context) -> bool:
        if self.value.t is None:
            mid = ctx.book.mid()
            if mid is None:
                return False
            self.value.start(ctx.t, mid)
        return True


class Bursty(InformedTrader):
    def __init__(
        self,
        value: Fundamental,
        rate: float,
        on: float,
        off: float,
        *,
        edge: float = 0.0,
        size: int = 1,
        seed: int = 0,
    ) -> None:
        super().__init__(value, rate, edge=edge, size=size, seed=seed)
        self.on = on
        self.off = off
        self.active = False
        self.switches: list[float] = []
        self._flip = math.inf
        self._switching = False

    def on_start(self, ctx: Context) -> None:
        self._begin(ctx)
        self._flip = ctx.t + self.rng.expovariate(self.on)
        self._next(ctx)

    def on_timer(self, ctx: Context) -> None:
        if self._switching:
            self.active = not self.active
            self.switches.append(ctx.t)
            self._flip = ctx.t + self.rng.expovariate(self.off if self.active else self.on)
        elif self._begin(ctx):
            self._trade(ctx)
        self._next(ctx)

    def _next(self, ctx: Context) -> None:
        dt = self._flip - ctx.t
        trade = self.rng.expovariate(self.rate) if self.active else math.inf
        self._switching = trade >= dt
        ctx.set_timer(min(trade, dt))
