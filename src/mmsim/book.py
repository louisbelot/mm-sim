"""Limit order book with price-time priority.

A pure-Python port of the matching rules in Lemypoo/orderbook, minus self-trade
prevention. Commands and the event stream are the same as in the C++ engine, so
the simulator can switch to it later without touching agent code.

Prices are integer ticks and quantities are integer lots.
"""

from __future__ import annotations

import dataclasses
from bisect import bisect_left, insort
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from enum import IntEnum


class Side(IntEnum):
    BUY = 0
    SELL = 1

    @property
    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY


class OrderType(IntEnum):
    LIMIT = 0
    MARKET = 1


class Tif(IntEnum):
    GTC = 0
    IOC = 1
    FOK = 2


class EventType(IntEnum):
    ACCEPTED = 0
    REJECTED = 1
    TRADE = 2
    CANCELED = 3
    MODIFIED = 4


class Reason(IntEnum):
    NONE = 0
    DUPLICATE_ID = 1
    UNKNOWN_ORDER = 2
    BAD_QTY = 3
    BAD_PRICE = 4
    BAD_FLAGS = 5  # market + post-only, or post-only that is not GTC
    USER = 6
    UNFILLED = 7  # IOC or market remainder
    FOK = 8
    POST_ONLY = 9


@dataclass(frozen=True, slots=True)
class Event:
    """One entry of the event stream. Field meaning by type:

    ACCEPTED  id, side, price, qty    (price 0 for market orders)
    REJECTED  id, reason              (side too, when a submit is rejected)
    TRADE     id = taker, maker, side = taker side, price = maker price, qty
    CANCELED  id, side, qty removed, reason
    MODIFIED  id, side, price, qty    (new price and leaves quantity)
    """

    type: EventType
    id: int
    side: Side = Side.BUY
    price: int = 0
    qty: int = 0
    maker: int = 0
    reason: Reason = Reason.NONE


@dataclass(slots=True)
class Order:
    id: int
    owner: int
    side: Side
    price: int
    qty: int
    post_only: bool = False


class Level:
    """Resting orders at one price, oldest first."""

    __slots__ = ("price", "qty", "orders")

    def __init__(self, price: int) -> None:
        self.price = price
        self.qty = 0
        self.orders: OrderedDict[int, Order] = OrderedDict()


# Levels are keyed by sign * price, so bids are stored negated. The best level
# on either side is then the smallest key, and matching is the same code for
# both sides: a taker crosses a level iff key <= sign * limit price.
_SIGN = (-1, 1)


class Book:
    def __init__(self) -> None:
        self._levels: tuple[dict[int, Level], dict[int, Level]] = ({}, {})
        self._keys: tuple[list[int], list[int]] = ([], [])  # sorted, best first
        self._orders: dict[int, Order] = {}

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
        """Enters a new order. The price is ignored for market orders."""
        if qty <= 0:
            return [Event(EventType.REJECTED, oid, side, reason=Reason.BAD_QTY)]
        if kind == OrderType.LIMIT and price <= 0:
            return [Event(EventType.REJECTED, oid, side, reason=Reason.BAD_PRICE)]
        if post_only and (kind == OrderType.MARKET or tif != Tif.GTC):
            return [Event(EventType.REJECTED, oid, side, reason=Reason.BAD_FLAGS)]
        if oid in self._orders:
            return [Event(EventType.REJECTED, oid, side, reason=Reason.DUPLICATE_ID)]

        if kind == OrderType.MARKET:
            price = 0
        out = [Event(EventType.ACCEPTED, oid, side, price, qty)]
        self._execute(Order(oid, owner, side, price, qty, post_only), kind, tif, out)
        return out

    def cancel(self, oid: int) -> list[Event]:
        o = self._orders.get(oid)
        if o is None:
            return [Event(EventType.REJECTED, oid, reason=Reason.UNKNOWN_ORDER)]
        self._remove(o)
        return [Event(EventType.CANCELED, oid, o.side, qty=o.qty, reason=Reason.USER)]

    def modify(self, oid: int, price: int, qty: int) -> list[Event]:
        """Sets a resting order's price and leaves quantity.

        Shrinking at the same price keeps time priority. Any other change sends the
        order to the back of the queue at its new price, where it can trade like a
        new GTC order that keeps its post-only flag.
        """
        if qty <= 0:
            return [Event(EventType.REJECTED, oid, reason=Reason.BAD_QTY)]
        if price <= 0:
            return [Event(EventType.REJECTED, oid, reason=Reason.BAD_PRICE)]
        o = self._orders.get(oid)
        if o is None:
            return [Event(EventType.REJECTED, oid, reason=Reason.UNKNOWN_ORDER)]

        out = [Event(EventType.MODIFIED, oid, o.side, price, qty)]
        if price == o.price and qty <= o.qty:
            self._levels[o.side][_SIGN[o.side] * price].qty -= o.qty - qty
            o.qty = qty
            return out
        self._remove(o)
        fresh = Order(oid, o.owner, o.side, price, qty, o.post_only)
        self._execute(fresh, OrderType.LIMIT, Tif.GTC, out)
        return out

    def best(self, side: Side) -> int | None:
        keys = self._keys[side]
        return _SIGN[side] * keys[0] if keys else None

    def best_bid(self) -> int | None:
        return self.best(Side.BUY)

    def best_ask(self) -> int | None:
        return self.best(Side.SELL)

    def mid(self) -> float | None:
        bid, ask = self.best_bid(), self.best_ask()
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2

    def levels(self, side: Side) -> Iterator[Level]:
        """Live levels of one side, best first. Read them, don't modify them."""
        levels = self._levels[side]
        return (levels[k] for k in self._keys[side])

    def level(self, side: Side, price: int) -> Level | None:
        return self._levels[side].get(_SIGN[side] * price)

    def depth(self, side: Side, n: int) -> list[tuple[int, int, int]]:
        """(price, qty, order count) for the n best levels."""
        out = []
        for lvl in self.levels(side):
            if len(out) == n:
                break
            out.append((lvl.price, lvl.qty, len(lvl.orders)))
        return out

    def orders(self, side: Side) -> list[Order]:
        """Copies of the resting orders, best price first, then time."""
        return [dataclasses.replace(o) for lvl in self.levels(side) for o in lvl.orders.values()]

    def find(self, oid: int) -> Order | None:
        o = self._orders.get(oid)
        return dataclasses.replace(o) if o is not None else None

    def ahead(self, oid: int) -> int:
        """Quantity in front of a resting order in its price queue."""
        o = self._orders[oid]
        qty = 0
        for other in self._levels[o.side][_SIGN[o.side] * o.price].orders.values():
            if other is o:
                break
            qty += other.qty
        return qty

    def __len__(self) -> int:
        return len(self._orders)

    def __contains__(self, oid: int) -> bool:
        return oid in self._orders

    def audit(self) -> str:
        """Cross-checks every structure. Returns the first broken invariant, or ""."""
        seen = 0
        for side in Side:
            keys, levels = self._keys[side], self._levels[side]
            if keys != sorted(set(keys)):
                return f"{side.name} keys not strictly sorted"
            if set(keys) != levels.keys():
                return f"{side.name} keys and levels disagree"
            for key in keys:
                lvl = levels[key]
                if lvl.price != _SIGN[side] * key:
                    return f"{side.name} level {lvl.price} stored under key {key}"
                if not lvl.orders:
                    return f"{side.name} level {lvl.price} is empty"
                if lvl.qty != sum(o.qty for o in lvl.orders.values()):
                    return f"{side.name} level {lvl.price} qty does not match its orders"
                for oid, o in lvl.orders.items():
                    if o.qty <= 0 or o.side != side or o.price != lvl.price:
                        return f"order {oid} is inconsistent with its level"
                    if self._orders.get(oid) is not o:
                        return f"order {oid} missing from the id index"
                    seen += 1
        if seen != len(self._orders):
            return "id index holds orders that are in no level"
        bid, ask = self.best_bid(), self.best_ask()
        if bid is not None and ask is not None and bid >= ask:
            return f"crossed book: bid {bid} >= ask {ask}"
        return ""

    def _execute(self, t: Order, kind: OrderType, tif: Tif, out: list[Event]) -> None:
        opp = 1 - t.side
        keys, levels = self._keys[opp], self._levels[opp]
        market = kind == OrderType.MARKET
        limit = _SIGN[opp] * t.price

        if t.post_only and keys and keys[0] <= limit:
            out.append(Event(EventType.CANCELED, t.id, t.side, qty=t.qty, reason=Reason.POST_ONLY))
            return
        if tif == Tif.FOK and not _fillable(keys, levels, limit, market, t.qty):
            out.append(Event(EventType.CANCELED, t.id, t.side, qty=t.qty, reason=Reason.FOK))
            return

        # Always trade against the head of the best level: each step either
        # removes that order or uses up the taker.
        while t.qty and keys and (market or keys[0] <= limit):
            lvl = levels[keys[0]]
            resting = lvl.orders
            while t.qty and resting:
                m = next(iter(resting.values()))
                q = min(t.qty, m.qty)
                out.append(Event(EventType.TRADE, t.id, t.side, lvl.price, q, maker=m.id))
                t.qty -= q
                m.qty -= q
                lvl.qty -= q
                if m.qty == 0:
                    del resting[m.id]
                    del self._orders[m.id]
            if not resting:
                del levels[keys[0]]
                del keys[0]

        if t.qty == 0:
            return
        if market or tif != Tif.GTC:
            out.append(Event(EventType.CANCELED, t.id, t.side, qty=t.qty, reason=Reason.UNFILLED))
        else:
            self._rest(t)

    def _rest(self, o: Order) -> None:
        key = _SIGN[o.side] * o.price
        levels = self._levels[o.side]
        lvl = levels.get(key)
        if lvl is None:
            lvl = levels[key] = Level(o.price)
            insort(self._keys[o.side], key)
        lvl.orders[o.id] = o
        lvl.qty += o.qty
        self._orders[o.id] = o

    def _remove(self, o: Order) -> None:
        key = _SIGN[o.side] * o.price
        levels = self._levels[o.side]
        lvl = levels[key]
        del lvl.orders[o.id]
        lvl.qty -= o.qty
        del self._orders[o.id]
        if not lvl.orders:
            del levels[key]
            keys = self._keys[o.side]
            del keys[bisect_left(keys, key)]


def _fillable(
    keys: list[int], levels: dict[int, Level], limit: int, market: bool, qty: int
) -> bool:
    """Read-only dry run of matching, so a killed fill-or-kill needs no rollback."""
    avail = 0
    for key in keys:
        if not market and key > limit:
            break
        avail += levels[key].qty
        if avail >= qty:
            return True
    return False
