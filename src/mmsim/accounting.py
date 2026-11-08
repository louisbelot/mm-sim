from __future__ import annotations

from collections.abc import Sequence
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


def exposure(fills: Sequence[Fill], start: float, end: float, q0: int = 0) -> float:
    area, t, q = 0.0, start, q0
    for f in fills:
        area += q * q * (f.t - t)
        t = f.t
        q += f.qty if f.side == Side.BUY else -f.qty
    return area + q * q * (end - t)
