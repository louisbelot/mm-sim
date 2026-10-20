"""Cash, inventory and mark-to-market P&L for one trader.

Units follow the book: prices in ticks, sizes in lots, so cash is in tick-lots.
Fees are ticks per lot, and a negative fee is a rebate.
"""

from __future__ import annotations

from dataclasses import dataclass

from .book import Side


@dataclass(frozen=True, slots=True)
class Fill:
    t: float
    oid: int
    side: Side  # our side of the trade
    price: int
    qty: int
    maker: bool  # our order was resting
    fee: float  # what we paid, negative for a rebate
    mid: float | None  # mid just before the trade, for markouts


class Account:
    def __init__(self, maker_fee: float = 0.0, taker_fee: float = 0.0) -> None:
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.cash = 0.0
        self.position = 0
        self.fees = 0.0
        self.volume = 0
        self.fills: list[Fill] = []

    def fill(
        self, t: float, oid: int, side: Side, price: int, qty: int, maker: bool, mid: float | None
    ) -> Fill:
        fee = (self.maker_fee if maker else self.taker_fee) * qty
        sign = 1 if side == Side.BUY else -1
        self.position += sign * qty
        self.cash -= sign * price * qty + fee
        self.fees += fee
        self.volume += qty
        f = Fill(t, oid, side, price, qty, maker, fee, mid)
        self.fills.append(f)
        return f

    def value(self, mark: float) -> float:
        """Mark-to-market P&L: cash plus the position valued at `mark`."""
        return self.cash + self.position * mark
