from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Iterator, Sequence

from .book import _SIGN, Book, Event, EventType, Order, OrderType, Reason, Side, Tif, _fillable
from .itch import NS_PER_MINUTE, Msg

MARKET = 0


class ReplayBook(Book):
    def best(self, side: Side) -> int | None:
        for lvl in self.levels(side):
            for o in lvl.orders.values():
                if o.owner == MARKET:
                    return lvl.price
        return None

    def apply(self, m: Msg, agg: tuple[int, int]) -> list[Event]:
        match m.kind:
            case "A" | "F":
                return self._add(m.ref, m.side, m.px, m.qty, agg[m.side])
            case "E" | "C":
                o = self._orders[-m.ref]
                side = o.side.opposite
                out = self._hit(agg[side], side, m.qty, o.price, o.id)
                if m.printable:
                    out.append(Event(EventType.TRADE, agg[side], side, o.price, m.qty, maker=o.id))
                self._shrink(o, m.qty)
                return out
            case "X":
                self._shrink(self._orders[-m.ref], m.qty)
            case "D":
                self._remove(self._orders[-m.ref])
            case "U":
                o = self._orders[-m.ref]
                self._remove(o)
                return self._add(m.new, o.side, m.px, m.qty, agg[o.side])
        return []

    def _add(self, ref: int, side: Side, px: int, qty: int, agg: int) -> list[Event]:
        out = self._hit(agg, side, qty, px)
        self._rest(Order(-ref, MARKET, side, px, qty))
        return out

    def _hit(self, agg: int, side: Side, qty: int, px: int, stop: int = 0) -> list[Event]:
        opp = side.opposite
        levels, limit = self._levels[opp], _SIGN[opp] * px
        targets = []
        for key in self._keys[opp]:
            if key > limit:
                break
            for o in levels[key].orders.values():
                if o.id == stop:
                    break
                if o.owner != MARKET:
                    targets.append(o)
        out = []
        for o in targets:
            if qty == 0:
                break
            q = min(qty, o.qty)
            out.append(Event(EventType.TRADE, agg, side, o.price, q, maker=o.id))
            qty -= q
            self._shrink(o, q)
        return out

    def _execute(self, t: Order, kind: OrderType, tif: Tif, out: list[Event]) -> None:
        opp = 1 - t.side
        keys, levels = self._keys[opp], self._levels[opp]
        market = kind == OrderType.MARKET
        limit = _SIGN[opp] * t.price
        if t.post_only or not keys or not (market or keys[0] <= limit):
            super()._execute(t, kind, tif, out)
            return
        if tif == Tif.FOK and not _fillable(keys, levels, limit, market, t.qty):
            out.append(Event(EventType.CANCELED, t.id, t.side, qty=t.qty, reason=Reason.FOK))
            return

        takes = []
        left = t.qty
        for key in keys:
            if left == 0 or not (market or key <= limit):
                break
            lvl = levels[key]
            for o in lvl.orders.values():
                if left == 0:
                    break
                q = min(left, o.qty)
                takes.append((o, lvl.price, q))
                left -= q
        for o, price, q in takes:
            out.append(Event(EventType.TRADE, t.id, t.side, price, q, maker=o.id))
            if o.owner != MARKET:
                self._shrink(o, q)
        t.qty = left
        if t.qty == 0:
            return
        if market or tif != Tif.GTC or (keys and keys[0] <= limit):
            out.append(Event(EventType.CANCELED, t.id, t.side, qty=t.qty, reason=Reason.UNFILLED))
        else:
            self._rest(t)

    def _shrink(self, o: Order, q: int) -> None:
        if q >= o.qty:
            self._remove(o)
        else:
            o.qty -= q
            self._levels[o.side][_SIGN[o.side] * o.price].qty -= q


class ItchFlow:
    owner = MARKET

    def __init__(self, msgs: Sequence[Msg], tick: int = 100) -> None:
        self.batches = batches(list(on_grid(msgs, tick)))
        self.t = 0.0
        self._i = 0

    def step(
        self, book: Book, rng: random.Random, horizon: float, ids: Iterator[int]
    ) -> tuple[float, Callable[[Book], list[Event]]] | None:
        if self._i < len(self.batches):
            batch = self.batches[self._i]
            dt = max(batch[0].t / NS_PER_MINUTE - self.t, 0.0)
            if dt <= horizon:
                self._i += 1
                self.t += dt
                agg = (next(ids), next(ids))
                return dt, lambda b: [e for m in batch for e in b.apply(m, agg)]
        self.t += horizon
        return None


def batches(msgs: list[Msg]) -> list[list[Msg]]:
    out: list[list[Msg]] = []
    for m in msgs:
        last = out[-1][-1] if out else None
        if last is not None and m.kind in "EC" and last.kind in "EC" and m.t == last.t:
            out[-1].append(m)
        else:
            out.append([m])
    return out


def on_grid(msgs: Iterable[Msg], tick: int) -> Iterator[Msg]:
    side: dict[int, Side] = {}
    off: set[int] = set()
    for m in msgs:
        if m.kind in "AFU":
            px, r = divmod(m.px, tick)
            was_off = m.ref in off
            if m.kind != "U":
                side[m.ref] = m.side
            else:
                side[m.new] = side.pop(m.ref)
                off.discard(m.ref)
            if r:
                off.add(m.new if m.kind == "U" else m.ref)
                if m.kind == "U" and not was_off:
                    yield Msg(m.t, "D", m.ref)
            elif m.kind == "U" and was_off:
                yield Msg(m.t, "A", m.new, side[m.new], px, m.qty)
            else:
                yield m._replace(px=px)
        elif m.ref not in off:
            yield m
