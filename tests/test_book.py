from mmsim.book import Book, Event, EventType, OrderType, Reason, Side, Tif

BUY, SELL = Side.BUY, Side.SELL
ACC, REJ, TRADE, CXL, MOD = (
    EventType.ACCEPTED,
    EventType.REJECTED,
    EventType.TRADE,
    EventType.CANCELED,
    EventType.MODIFIED,
)


def book_with(*orders):
    b = Book()
    for oid, side, px, qty in orders:
        b.submit(oid, side, px, qty)
    return b


def trades(events):
    return [(e.id, e.maker, e.price, e.qty) for e in events if e.type == TRADE]


def test_rests_when_not_crossing():
    b = book_with((1, BUY, 99, 5), (2, SELL, 101, 3))
    assert b.best_bid() == 99 and b.best_ask() == 101
    assert b.mid() == 100
    assert b.depth(BUY, 5) == [(99, 5, 1)]
    assert b.audit() == ""


def test_price_then_time_priority():
    b = book_with((1, SELL, 101, 2), (2, SELL, 100, 2), (3, SELL, 100, 2))
    ev = b.submit(4, BUY, 101, 5)
    assert trades(ev) == [(4, 2, 100, 2), (4, 3, 100, 2), (4, 1, 101, 1)]
    assert b.find(1).qty == 1
    assert b.best_bid() is None


def test_trade_prints_at_maker_price():
    b = book_with((1, BUY, 100, 4))
    ev = b.submit(2, SELL, 95, 1)
    assert trades(ev) == [(2, 1, 100, 1)]
    assert ev[1].side == SELL


def test_limit_remainder_rests_after_sweep():
    b = book_with((1, SELL, 100, 2))
    b.submit(2, BUY, 101, 5)
    assert b.best_bid() == 101 and b.find(2).qty == 3
    assert b.best_ask() is None


def test_market_order_sweeps_and_cancels_rest():
    b = book_with((1, SELL, 100, 2), (2, SELL, 102, 1))
    ev = b.submit(3, BUY, 0, 5, kind=OrderType.MARKET)
    assert ev[0] == Event(ACC, 3, BUY, 0, 5)
    assert trades(ev) == [(3, 1, 100, 2), (3, 2, 102, 1)]
    assert ev[-1] == Event(CXL, 3, BUY, qty=2, reason=Reason.UNFILLED)
    assert len(b) == 0


def test_ioc_never_rests():
    b = book_with((1, SELL, 100, 2))
    ev = b.submit(2, BUY, 100, 3, tif=Tif.IOC)
    assert trades(ev) == [(2, 1, 100, 2)]
    assert ev[-1].reason == Reason.UNFILLED and ev[-1].qty == 1
    assert b.best_bid() is None


def test_fok_kill_leaves_book_untouched():
    b = book_with((1, SELL, 100, 2), (2, SELL, 101, 2), (3, SELL, 103, 9))
    ev = b.submit(4, BUY, 101, 5, tif=Tif.FOK)
    assert ev[-1] == Event(CXL, 4, BUY, qty=5, reason=Reason.FOK)
    assert b.depth(SELL, 3) == [(100, 2, 1), (101, 2, 1), (103, 9, 1)]


def test_fok_fills_exactly_across_levels():
    b = book_with((1, SELL, 100, 2), (2, SELL, 101, 3))
    ev = b.submit(4, BUY, 101, 5, tif=Tif.FOK)
    assert trades(ev) == [(4, 1, 100, 2), (4, 2, 101, 3)]
    assert len(b) == 0


def test_post_only_cancels_instead_of_crossing():
    b = book_with((1, SELL, 100, 2))
    ev = b.submit(2, BUY, 100, 1, post_only=True)
    assert ev[-1] == Event(CXL, 2, BUY, qty=1, reason=Reason.POST_ONLY)
    ev = b.submit(3, BUY, 99, 1, post_only=True)
    assert [e.type for e in ev] == [ACC]
    assert b.best_bid() == 99


def test_rejections():
    b = book_with((1, BUY, 99, 1))
    assert b.submit(2, BUY, 99, 0)[0].reason == Reason.BAD_QTY
    assert b.submit(2, BUY, 0, 1)[0].reason == Reason.BAD_PRICE
    assert b.submit(1, SELL, 101, 1)[0].reason == Reason.DUPLICATE_ID
    assert b.submit(2, BUY, 0, 1, kind=OrderType.MARKET, post_only=True)[0].reason == (
        Reason.BAD_FLAGS
    )
    assert b.submit(2, BUY, 98, 1, tif=Tif.IOC, post_only=True)[0].reason == Reason.BAD_FLAGS
    assert b.cancel(7) == [Event(REJ, 7, reason=Reason.UNKNOWN_ORDER)]
    assert b.modify(7, 99, 1)[0].reason == Reason.UNKNOWN_ORDER
    assert b.modify(1, 99, 0)[0].reason == Reason.BAD_QTY
    assert len(b) == 1 and b.find(1).qty == 1


def test_filled_id_can_be_reused():
    b = book_with((1, SELL, 100, 1))
    b.submit(2, BUY, 100, 1)
    assert [e.type for e in b.submit(1, BUY, 99, 1)] == [ACC]


def test_cancel_removes_order_and_empty_level():
    b = book_with((1, BUY, 99, 2), (2, BUY, 98, 1))
    assert b.cancel(1) == [Event(CXL, 1, BUY, qty=2, reason=Reason.USER)]
    assert b.best_bid() == 98
    assert b.level(BUY, 99) is None
    assert b.audit() == ""


def test_modify_size_down_keeps_priority():
    b = book_with((1, BUY, 99, 5), (2, BUY, 99, 5))
    assert b.modify(1, 99, 2) == [Event(MOD, 1, BUY, 99, 2)]
    assert b.ahead(2) == 2
    ev = b.submit(3, SELL, 99, 2)
    assert trades(ev) == [(3, 1, 99, 2)]


def test_modify_size_up_loses_priority():
    b = book_with((1, BUY, 99, 5), (2, BUY, 99, 5))
    b.modify(1, 99, 6)
    assert b.ahead(1) == 5 and b.ahead(2) == 0


def test_modify_reprice_can_trade():
    b = book_with((1, BUY, 99, 5), (2, SELL, 101, 2))
    ev = b.modify(1, 101, 5)
    assert trades(ev) == [(1, 2, 101, 2)]
    assert b.find(1).qty == 3 and b.best_bid() == 101


def test_modify_keeps_post_only():
    b = Book()
    b.submit(1, BUY, 99, 5, post_only=True)
    b.submit(2, SELL, 101, 2)
    ev = b.modify(1, 101, 5)
    assert ev[-1] == Event(CXL, 1, BUY, qty=5, reason=Reason.POST_ONLY)
    assert 1 not in b


def test_ahead_counts_quantity_not_orders():
    b = book_with((1, SELL, 100, 3), (2, SELL, 100, 4), (3, SELL, 100, 1))
    assert [b.ahead(i) for i in (1, 2, 3)] == [0, 3, 7]


def test_orders_are_copies():
    b = book_with((1, BUY, 99, 5))
    b.orders(BUY)[0].qty = 1
    b.find(1).qty = 2
    assert b.find(1).qty == 5
