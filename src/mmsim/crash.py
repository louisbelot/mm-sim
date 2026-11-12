from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from .book import Book, Event, EventType, OrderType, Reason, Side, Tif


@dataclass(frozen=True)
class Rule:
    band: float | None = None
    limit: float | None = None
    trip: float | None = None
    pause: float = 5.0
    window: float = 5.0

    def __post_init__(self) -> None:
        if self.limit is not None and self.band is None:
            raise ValueError("a limit state needs a band")


class Guard(Book):
    def __init__(self, rule: Rule | None = None) -> None:
        super().__init__()
        self.rule = rule or Rule()
        self.clock: Callable[[], float] = lambda: 0.0
        self.ref = math.nan
        self.trades: list[tuple[float, int, int]] = []
        self.pauses: list[tuple[float, float]] = []
        self._window: deque[tuple[float, int]] = deque()
        self._sum = 0
        self._since: float | None = None
        self._paused = False
        self._until = -math.inf

    def halted(self) -> bool:
        self._tick(self.clock())
        return self._paused

    def bands(self) -> tuple[int, int] | None:
        if self.rule.band is None or math.isnan(self.ref):
            return None
        return math.ceil(self.ref - self.rule.band), math.floor(self.ref + self.rule.band)

    def submit(
        self,
        oid: int,
        side: Side,
        price: int,
        qty: int,
        *,
        owner: int = 0,
        kind: OrderType = OrderType.LIMIT,
        tif: Tif = Tif.GTC,
        post_only: bool = False,
    ) -> list[Event]:
        now = self._enter()
        if self._paused and self._marketable(side, price, kind):
            return [Event(EventType.REJECTED, oid, side, reason=Reason.HALTED)]
        b = self.bands()
        if b is not None and not post_only:
            lo, hi = b
            if kind == OrderType.MARKET:
                kind, price = OrderType.LIMIT, lo if side == Side.SELL else hi
                tif = Tif.FOK if tif == Tif.FOK else Tif.IOC
            else:
                price = max(price, lo) if side == Side.SELL else min(price, hi)
        out = super().submit(
            oid, side, price, qty, owner=owner, kind=kind, tif=tif, post_only=post_only
        )
        self._leave(now, out)
        return out

    def modify(self, oid: int, price: int, qty: int) -> list[Event]:
        now = self._enter()
        o = self._orders.get(oid)
        if o is not None:
            if self._paused and self._marketable(o.side, price, OrderType.LIMIT):
                return [Event(EventType.REJECTED, oid, o.side, reason=Reason.HALTED)]
            b = self.bands()
            if b is not None:
                price = max(price, b[0]) if o.side == Side.SELL else min(price, b[1])
        out = super().modify(oid, price, qty)
        self._leave(now, out)
        return out

    def cancel(self, oid: int) -> list[Event]:
        now = self._enter()
        out = super().cancel(oid)
        self._leave(now, out)
        return out

    def _enter(self) -> float:
        now = self.clock()
        self._tick(now)
        if self._since is None and not self._paused:
            w = self._window
            while w and w[0][0] < now - self.rule.window:
                self._sum -= w.popleft()[1]
            if w:
                self.ref = self._sum / len(w)
            elif math.isnan(self.ref):
                mid = self.mid()
                if mid is not None:
                    self.ref = mid
        return now

    def _leave(self, now: float, out: list[Event]) -> None:
        r = self.rule
        prints = [e for e in out if e.type == EventType.TRADE]
        for e in prints:
            self.trades.append((now, e.price, e.qty))
            self._window.append((now, e.price))
            self._sum += e.price
        if self._paused or math.isnan(self.ref):
            return
        if r.trip is not None and any(abs(e.price - self.ref) >= r.trip for e in prints):
            self._halt(now)
            return
        if r.limit is not None:
            lo, hi = self.bands()
            bid, ask = self.best_bid(), self.best_ask()
            stuck = bid is None or bid < lo or ask is None or ask > hi
            if not stuck:
                self._since = None
            elif self._since is None:
                self._since = now

    def _tick(self, now: float) -> None:
        if self._since is not None and now >= self._since + self.rule.limit:
            self._halt(self._since + self.rule.limit)
        if self._paused and now >= self._until:
            self._paused = False
            self.ref = math.nan
            self._window.clear()
            self._sum = 0

    def _halt(self, t: float) -> None:
        self._since = None
        self._paused = True
        self._until = t + self.rule.pause
        self.pauses.append((t, self._until))

    def _marketable(self, side: Side, price: int, kind: OrderType) -> bool:
        if kind == OrderType.MARKET:
            return True
        if side == Side.BUY:
            ask = self.best_ask()
            return ask is not None and price >= ask
        bid = self.best_bid()
        return bid is not None and price <= bid
