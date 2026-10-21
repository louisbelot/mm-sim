import random

import pytest
from reference_book import ReferenceBook

from mmsim.book import Book, OrderType, Side, Tif


def commands(rng, ref, n):
    next_id = 1
    for _ in range(n):
        ids = ref.ids()
        r = rng.random()
        if ids and r < 0.15:
            yield ("cancel", rng.choice(ids) if rng.random() < 0.95 else 10**9)
        elif ids and r < 0.30:
            o = ref._find(rng.choice(ids))
            u = rng.random()
            if u < 0.4:
                px, qty = o.price, rng.randint(1, o.qty)
            elif u < 0.6:
                px, qty = o.price, rng.randint(0, 9)
            else:
                px, qty = max(0, o.price + rng.randint(-3, 3)), rng.randint(1, 8)
            yield ("modify", o.id, px, qty)
        else:
            side = rng.choice((Side.BUY, Side.SELL))
            centre = 99 if side == Side.BUY else 101
            px = centre + rng.randint(-3, 3)
            if rng.random() < 0.01:
                px = 0
            qty = rng.randint(1, 8) if rng.random() > 0.02 else 0
            kind = OrderType.MARKET if rng.random() < 0.08 else OrderType.LIMIT
            tif = rng.choices((Tif.GTC, Tif.IOC, Tif.FOK), (70, 15, 15))[0]
            post_only = rng.random() < 0.15
            oid = rng.choice(ids) if ids and rng.random() < 0.02 else next_id
            next_id += 1
            yield ("submit", oid, side, px, qty, rng.randint(0, 2), kind, tif, post_only)


def apply(book, cmd):
    match cmd:
        case ("cancel", oid):
            return book.cancel(oid)
        case ("modify", oid, px, qty):
            return book.modify(oid, px, qty)
        case ("submit", oid, side, px, qty, owner, kind, tif, post_only):
            return book.submit(
                oid, side, px, qty, owner=owner, kind=kind, tif=tif, post_only=post_only
            )


def snapshot(book, side):
    return [(o.id, o.owner, o.side, o.price, o.qty) for o in book.orders(side)]


@pytest.mark.parametrize("seed", range(30))
def test_matches_reference(seed):
    rng = random.Random(seed)
    book, ref = Book(), ReferenceBook()
    for i, cmd in enumerate(commands(rng, ref, 2000)):
        got, want = apply(book, cmd), apply(ref, cmd)
        assert got == want, f"command {i}: {cmd}"
        for side in Side:
            assert snapshot(book, side) == ref.snapshot(side), f"command {i}: {cmd}"
        assert book.audit() == "", f"command {i}: {cmd}"
