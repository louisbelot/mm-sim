from __future__ import annotations

import math

from . import avellaneda, glft
from .accounting import Fill
from .book import Side
from .sim import Agent, Context


class Quoter(Agent):
    def __init__(self, size: int = 1, max_position: int | None = None) -> None:
        self.size = size
        self.max_position = max_position
        self._oids: dict[Side, int | None] = {Side.BUY: None, Side.SELL: None}

    def targets(self, ctx: Context) -> tuple[int | None, int | None]:
        raise NotImplementedError

    def on_start(self, ctx: Context) -> None:
        self.requote(ctx)

    def on_book(self, ctx: Context) -> None:
        self.requote(ctx)

    def on_fill(self, ctx: Context, fill: Fill) -> None:
        self.requote(ctx)

    def requote(self, ctx: Context) -> None:
        bid, ask = self.targets(ctx)
        if self.max_position is not None:
            if ctx.position + self.size > self.max_position:
                bid = None
            if ctx.position - self.size < -self.max_position:
                ask = None
        self._quote(ctx, Side.BUY, bid)
        self._quote(ctx, Side.SELL, ask)

    def _quote(self, ctx: Context, side: Side, px: int | None) -> None:
        oid = self._oids[side]
        live = ctx.orders.get(oid)
        if live is not None and live.price == px:
            return
        if live is not None:
            ctx.cancel(oid)
        self._oids[side] = None if px is None else ctx.submit(side, px, self.size, post_only=True)


class SymmetricQuoter(Quoter):
    def __init__(self, half_spread: float = 0.5, size: int = 1, max_position: int | None = None):
        super().__init__(size, max_position)
        self.half_spread = half_spread

    def targets(self, ctx: Context) -> tuple[int | None, int | None]:
        mid = ctx.mid
        if mid is None:
            return None, None
        return math.floor(mid - self.half_spread), math.ceil(mid + self.half_spread)


class AvellanedaStoikov(Quoter):
    def __init__(
        self,
        gamma: float,
        sigma: float,
        k: float,
        *,
        horizon: float | None = None,
        size: int = 1,
        max_position: int | None = None,
    ) -> None:
        super().__init__(size, max_position)
        self.gamma = gamma
        self.sigma = sigma
        self.k = k
        self.horizon = horizon

    def targets(self, ctx: Context) -> tuple[int | None, int | None]:
        s = ctx.mid
        if s is None:
            return None, None
        tau = ctx.end - ctx.t
        if self.horizon is not None:
            tau = min(tau, self.horizon)
        bid, ask = avellaneda.quotes(s, ctx.position, tau, self.gamma, self.sigma, self.k)
        return min(math.floor(bid), ctx.ask - 1), max(math.ceil(ask), ctx.bid + 1)


class GLFT(Quoter):
    def __init__(self, model: glft.Model, *, size: int = 1) -> None:
        super().__init__(size, model.Q * size)
        self.model = model
        self.bids, self.asks = model.quotes(math.inf)

    def targets(self, ctx: Context) -> tuple[int | None, int | None]:
        s = ctx.mid
        if s is None:
            return None, None
        Q = self.model.Q
        i = Q + min(max(round(ctx.position / self.size), -Q), Q)
        db, da = self.bids[i], self.asks[i]
        bid = None if math.isnan(db) else min(math.floor(s - db), ctx.ask - 1)
        ask = None if math.isnan(da) else max(math.ceil(s + da), ctx.bid + 1)
        return bid, ask
