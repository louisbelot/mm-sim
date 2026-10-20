"""Market-making bots."""

from __future__ import annotations

import math

from .accounting import Fill
from .book import Side
from .sim import Agent, Context


class Quoter(Agent):
    """Keeps one post-only order on each side, at the prices `targets` picks.

    A quote whose target has not moved is left alone, so it keeps its place in
    the queue. A side is dropped while a fill there would take |position| past
    max_position. Subclasses only choose prices.
    """

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
    """Bid at floor(mid - half_spread), ask at ceil(mid + half_spread).

    The mid is everyone else's, so the quotes follow the market instead of the
    bot's own orders. half_spread = 0.5 joins the best bid and ask.
    """

    def __init__(self, half_spread: float = 0.5, size: int = 1, max_position: int | None = None):
        super().__init__(size, max_position)
        self.half_spread = half_spread

    def targets(self, ctx: Context) -> tuple[int | None, int | None]:
        mid = ctx.mid
        if mid is None:
            return None, None
        return math.floor(mid - self.half_spread), math.ceil(mid + self.half_spread)
