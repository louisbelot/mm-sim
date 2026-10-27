from __future__ import annotations

from orderbook import Book as Engine

from .book import Book, Event, EventType, Order, OrderType, Reason, Side, Tif

_SIDES = ("buy", "sell")
_TIFS = ("gtc", "ioc", "fok")


class NativeBook(Book):
    def __init__(self) -> None:
        super().__init__()
        self._engine = Engine()

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
        market = kind == OrderType.MARKET
        raw = self._engine.submit(
            oid,
            _SIDES[side],
            qty,
            None if market else price,
            owner=owner,
            tif=_TIFS[tif],
            post_only=post_only,
        )
        return self._apply(_convert(raw, market), owner, post_only)

    def cancel(self, oid: int) -> list[Event]:
        return self._apply(_convert(self._engine.cancel(oid)), 0, False)

    def modify(self, oid: int, price: int, qty: int) -> list[Event]:
        o = self._orders.get(oid)
        raw = self._engine.modify(oid, price, qty)
        return self._apply(_convert(raw), o.owner if o else 0, o.post_only if o else False)

    def audit(self) -> str:
        if msg := super().audit():
            return msg
        if msg := self._engine.audit():
            return f"engine: {msg}"
        if len(self._engine) != len(self):
            return f"engine holds {len(self._engine)} orders, the view {len(self)}"
        for oid, o in self._orders.items():
            theirs = self._engine.find(oid)
            if theirs is None or (theirs.owner, theirs.price, theirs.qty) != (
                o.owner,
                o.price,
                o.qty,
            ):
                return f"order {oid} differs from the engine"
        for side in Side:
            n = len(self._keys[side])
            theirs = [(lv.price, lv.qty, lv.count) for lv in self._engine.depth(_SIDES[side], n)]
            if theirs != self.depth(side, n):
                return f"{side.name} depth differs from the engine"
        return ""

    def __deepcopy__(self, memo: dict) -> NativeBook:
        twin = NativeBook()
        for side in Side:
            for o in self.orders(side):
                twin.submit(o.id, o.side, o.price, o.qty, owner=o.owner, post_only=o.post_only)
        memo[id(self)] = twin
        return twin

    def _apply(self, events: list[Event], owner: int, post_only: bool) -> list[Event]:
        pending: Order | None = None
        for e in events:
            match e.type:
                case EventType.ACCEPTED:
                    pending = Order(e.id, owner, e.side, e.price, e.qty, post_only)
                case EventType.MODIFIED:
                    o = self._orders[e.id]
                    if e.price == o.price and e.qty <= o.qty:
                        self._levels[o.side][_key(o)].qty -= o.qty - e.qty
                        o.qty = e.qty
                    else:
                        self._remove(o)
                        pending = Order(e.id, o.owner, o.side, e.price, e.qty, o.post_only)
                case EventType.TRADE:
                    maker = self._orders[e.maker]
                    self._levels[maker.side][_key(maker)].qty -= e.qty
                    maker.qty -= e.qty
                    if maker.qty == 0:
                        self._remove(maker)
                    if pending is not None and pending.id == e.id:
                        pending.qty -= e.qty
                case EventType.CANCELED:
                    if pending is not None and pending.id == e.id:
                        pending = None
                    elif e.id in self._orders:
                        self._remove(self._orders[e.id])
        if pending is not None and pending.qty > 0:
            self._rest(pending)
        return events


def _key(o: Order) -> int:
    return (-1, 1)[o.side] * o.price


def _convert(raw: list, market: bool = False) -> list[Event]:
    out = []
    for e in raw:
        side = Side.BUY if e.side == "buy" else Side.SELL
        price = 0 if market and e.type == "accepted" else e.price
        out.append(
            Event(
                EventType[e.type.upper()],
                e.id,
                side,
                price,
                e.qty,
                e.maker,
                Reason[e.reason.upper()],
            )
        )
    return out
