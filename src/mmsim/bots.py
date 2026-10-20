"""Market-making bots."""

from __future__ import annotations

import math

from . import avellaneda
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


class AvellanedaStoikov(Quoter):
    """Avellaneda-Stoikov quotes on the live book.

    Quotes the paper's spread around r = s - q gamma sigma^2 tau, with q the
    position in lots and tau the time left in the run, or `horizon` if that is
    shorter. Units are ticks, lots and the flow's time unit. Model prices are
    rounded away from r to whole ticks and then kept passive: in the model a
    quote can sit through the mid, but on a book that would be a market order,
    so a bid stops one tick below everyone else's ask (and the ask one tick
    above their bid).
    """

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
