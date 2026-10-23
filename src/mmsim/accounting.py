from __future__ import annotations

from dataclasses import dataclass

from .book import Side


@dataclass(frozen=True, slots=True)
class Fill:
    t: float
    oid: int
    side: Side
    price: int
    qty: int
    maker: bool
    fee: float
    mid: float | None
    counterparty: int = 0
    ahead: int | None = None
    left: int | None = None
    stale: bool = False


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
        self,
        t: float,
        oid: int,
        side: Side,
        price: int,
        qty: int,
        maker: bool,
        mid: float | None,
        counterparty: int = 0,
        ahead: int | None = None,
        left: int | None = None,
        stale: bool = False,
    ) -> Fill:
        fee = (self.maker_fee if maker else self.taker_fee) * qty
        sign = 1 if side == Side.BUY else -1
        self.position += sign * qty
        self.cash -= sign * price * qty + fee
        self.fees += fee
        self.volume += qty
        f = Fill(t, oid, side, price, qty, maker, fee, mid, counterparty, ahead, left, stale)
        self.fills.append(f)
        return f

    def value(self, mark: float) -> float:
        return self.cash + self.position * mark
