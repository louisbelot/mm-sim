"""Deliberately naive order book, used as the oracle in the differential test.

It shares only the event types with mmsim.book. Resting orders sit in one flat
list, and every take sorts the crossing orders from scratch.
"""

from dataclasses import dataclass

from mmsim.book import Event, EventType, OrderType, Reason, Side, Tif


@dataclass
class RefOrder:
    id: int
    owner: int
    side: Side
    price: int
    qty: int
    post_only: bool
    seq: int


class ReferenceBook:
    def __init__(self):
        self.resting: list[RefOrder] = []
        self.seq = 0

    def submit(
        self, oid, side, price, qty, *, owner=0, kind=OrderType.LIMIT, tif=Tif.GTC, post_only=False
    ):
        if qty <= 0:
            return [Event(EventType.REJECTED, oid, side, reason=Reason.BAD_QTY)]
        if kind == OrderType.LIMIT and price <= 0:
            return [Event(EventType.REJECTED, oid, side, reason=Reason.BAD_PRICE)]
        if post_only and (kind == OrderType.MARKET or tif != Tif.GTC):
            return [Event(EventType.REJECTED, oid, side, reason=Reason.BAD_FLAGS)]
        if self._find(oid):
            return [Event(EventType.REJECTED, oid, side, reason=Reason.DUPLICATE_ID)]
        if kind == OrderType.MARKET:
            price = 0
        out = [Event(EventType.ACCEPTED, oid, side, price, qty)]
        self._take(oid, owner, side, price, qty, kind, tif, post_only, out)
        return out

    def cancel(self, oid):
        o = self._find(oid)
        if o is None:
            return [Event(EventType.REJECTED, oid, reason=Reason.UNKNOWN_ORDER)]
        self.resting.remove(o)
        return [Event(EventType.CANCELED, oid, o.side, qty=o.qty, reason=Reason.USER)]

    def modify(self, oid, price, qty):
        if qty <= 0:
            return [Event(EventType.REJECTED, oid, reason=Reason.BAD_QTY)]
        if price <= 0:
            return [Event(EventType.REJECTED, oid, reason=Reason.BAD_PRICE)]
        o = self._find(oid)
        if o is None:
            return [Event(EventType.REJECTED, oid, reason=Reason.UNKNOWN_ORDER)]
        out = [Event(EventType.MODIFIED, oid, o.side, price, qty)]
        if price == o.price and qty <= o.qty:
            o.qty = qty
            return out
        self.resting.remove(o)
        self._take(oid, o.owner, o.side, price, qty, OrderType.LIMIT, Tif.GTC, o.post_only, out)
        return out

    def snapshot(self, side):
        sign = -1 if side == Side.BUY else 1
        mine = sorted(
            (o for o in self.resting if o.side == side), key=lambda o: (sign * o.price, o.seq)
        )
        return [(o.id, o.owner, o.side, o.price, o.qty) for o in mine]

    def ids(self):
        return [o.id for o in self.resting]

    def _find(self, oid):
        return next((o for o in self.resting if o.id == oid), None)

    def _take(self, oid, owner, side, price, qty, kind, tif, post_only, out):
        market = kind == OrderType.MARKET

        def crosses(o):
            if o.side == side:
                return False
            if market:
                return True
            return o.price <= price if side == Side.BUY else o.price >= price

        sign = 1 if side == Side.BUY else -1
        makers = sorted(
            (o for o in self.resting if crosses(o)), key=lambda o: (sign * o.price, o.seq)
        )
        if post_only and makers:
            out.append(Event(EventType.CANCELED, oid, side, qty=qty, reason=Reason.POST_ONLY))
            return
        if tif == Tif.FOK and sum(o.qty for o in makers) < qty:
            out.append(Event(EventType.CANCELED, oid, side, qty=qty, reason=Reason.FOK))
            return
        for m in makers:
            if qty == 0:
                break
            q = min(qty, m.qty)
            out.append(Event(EventType.TRADE, oid, side, m.price, q, maker=m.id))
            qty -= q
            m.qty -= q
        self.resting = [o for o in self.resting if o.qty > 0]
        if qty == 0:
            return
        if market or tif != Tif.GTC:
            out.append(Event(EventType.CANCELED, oid, side, qty=qty, reason=Reason.UNFILLED))
            return
        self.seq += 1
        self.resting.append(RefOrder(oid, owner, side, price, qty, post_only, self.seq))
