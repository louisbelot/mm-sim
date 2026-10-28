import struct
from pathlib import Path

import pytest

from mmsim.book import Side
from mmsim.itch import Msg, decode, extract, frames, read

SESSION = Path(__file__).parent / "data" / "session.itch"


def frame(body: bytes) -> bytes:
    return struct.pack(">H", len(body)) + body


def header(kind: bytes, locate: int, t: int) -> bytes:
    return kind + struct.pack(">HHHI", locate, 0, t >> 32, t & 0xFFFFFFFF)


def directory(locate: int, symbol: str) -> bytes:
    return frame(header(b"R", locate, 0) + symbol.encode().ljust(8) + bytes(20))


def add(locate: int, t: int, ref: int, side: bytes, qty: int, symbol: str, px: int) -> bytes:
    body = struct.pack(">QcI8sI", ref, side, qty, symbol.encode().ljust(8), px)
    return frame(header(b"A", locate, t) + body)


def test_session_written_by_the_cpp_engine():
    msgs = read(SESSION)
    kinds = "".join(m.kind for m in msgs)
    assert kinds == "AAAAAAEEXUEEEDAEEAAA"
    assert msgs[0] == Msg(34_200_001_000_000, "A", 1, Side.SELL, 10100, 300)
    assert msgs[9] == Msg(34_200_009_000_000, "U", 6, px=10000, qty=100, new=7)
    assert sum(m.qty for m in msgs if m.kind == "E") == 1200


def test_timestamp_uses_all_six_bytes():
    t = 57_599_999_999_999
    (m,) = [decode(f) for f in frames(add(1, t, 9, b"B", 100, "X", 10000))]
    assert m.t == t and m.ref == 9 and m.side == Side.BUY


def test_frames_reject_truncated_input():
    raw = add(1, 0, 1, b"B", 1, "X", 100)
    with pytest.raises(ValueError, match="truncated"):
        list(frames(raw[:-1]))
    with pytest.raises(ValueError, match="trailing"):
        list(frames(raw + b"\x00"))


def test_decode_checks_length_and_skips_unknown_types():
    assert decode(header(b"I", 1, 0) + bytes(39)) is None
    with pytest.raises(ValueError, match="A message of 35"):
        decode(header(b"A", 1, 0) + bytes(24))


def test_extract_keeps_one_locate(tmp_path):
    raw = b"".join(
        [
            directory(1, "AAA"),
            directory(2, "BBB"),
            add(1, 5, 1, b"B", 100, "AAA", 10000),
            add(2, 6, 2, b"S", 200, "BBB", 20000),
            add(1, 7, 3, b"S", 300, "AAA", 10100),
        ]
    )
    src = tmp_path / "day.itch"
    src.write_bytes(raw)
    paths = extract(src, ["BBB"], tmp_path / "out")
    assert paths["BBB"].read_bytes() == directory(2, "BBB") + add(2, 6, 2, b"S", 200, "BBB", 20000)
    assert [m.ref for m in read(paths["BBB"])] == [2]
