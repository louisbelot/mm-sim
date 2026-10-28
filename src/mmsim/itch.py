from __future__ import annotations

import struct
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

from .book import Side

NS_PER_MINUTE = 60_000_000_000

_LAYOUT = {
    b"A": struct.Struct(">cHHHIQcI8sI"),
    b"F": struct.Struct(">cHHHIQcI8sI4s"),
    b"E": struct.Struct(">cHHHIQIQ"),
    b"C": struct.Struct(">cHHHIQIQcI"),
    b"X": struct.Struct(">cHHHIQI"),
    b"D": struct.Struct(">cHHHIQ"),
    b"U": struct.Struct(">cHHHIQQII"),
    b"P": struct.Struct(">cHHHIQcI8sIQ"),
    b"Q": struct.Struct(">cHHHIQ8sIQc"),
}
_SIDE = {b"B": Side.BUY, b"S": Side.SELL}


class Msg(NamedTuple):
    t: int
    kind: str
    ref: int
    side: Side = Side.BUY
    px: int = 0
    qty: int = 0
    new: int = 0
    printable: bool = True


def frames(buf) -> Iterator[memoryview]:
    view = memoryview(buf)
    pos, n = 0, len(view)
    while n - pos >= 2:
        ln = view[pos] << 8 | view[pos + 1]
        end = pos + 2 + ln
        if end > n:
            raise ValueError(f"truncated message at byte {pos}")
        yield view[pos + 2 : end]
        pos = end
    if pos != n:
        raise ValueError(f"trailing byte at {pos}")


def decode(m) -> Msg | None:
    layout = _LAYOUT.get(bytes(m[:1]))
    if layout is None:
        return None
    if len(m) != layout.size:
        raise ValueError(f"{bytes(m[:1]).decode()} message of {len(m)} bytes")
    f = layout.unpack(m)
    kind, t = f[0].decode(), f[3] << 32 | f[4]
    match kind:
        case "A" | "F":
            return Msg(t, kind, f[5], _SIDE[f[6]], f[9], f[7])
        case "E":
            return Msg(t, kind, f[5], qty=f[6])
        case "C":
            return Msg(t, kind, f[5], px=f[9], qty=f[6], printable=f[8] == b"Y")
        case "X":
            return Msg(t, kind, f[5], qty=f[6])
        case "D":
            return Msg(t, kind, f[5])
        case "U":
            return Msg(t, kind, f[5], px=f[8], qty=f[7], new=f[6])
        case "P":
            return Msg(t, kind, f[5], _SIDE[f[6]], f[9], f[7])
        case _:
            return Msg(t, kind, 0, px=f[7], qty=f[5])


def read(path: str | Path) -> list[Msg]:
    data = Path(path).read_bytes()
    return [msg for m in frames(data) if (msg := decode(m)) is not None]
