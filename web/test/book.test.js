import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { test } from "node:test"

import { BUY, Book, SELL } from "../js/book.js"
import { AvellanedaStoikov, GLFT } from "../js/bots.js"
import { ZeroIntelligence, skyPerfect } from "../js/cst.js"
import { Model } from "../js/glft.js"
import { Market, session } from "../js/market.js"
import { mulberry32 } from "../js/rng.js"
import { mean, sd } from "../js/stats.js"

const ref = JSON.parse(readFileSync(new URL("./book_reference.json", import.meta.url)))

const close = (got, want, tol, what) =>
  assert.ok(Math.abs(got - want) <= tol, `${what}: got ${got}, want ${want} +- ${tol}`)

const depth = (book, side) => [...book.levels(side)].map((l) => [l.price, l.qty])

class Naive {
  constructor() {
    this.orders = []
    this.seq = 0
  }

  submit(id, side, price, qty, market, postOnly) {
    const better = (o) => (side === BUY ? o.price <= price : o.price >= price)
    const opp = this.orders
      .filter((o) => o.side !== side && (market || better(o)))
      .sort((a, b) => (side === BUY ? a.price - b.price : b.price - a.price) || a.seq - b.seq)
    if (postOnly && opp.length) return []
    const trades = []
    for (const o of opp) {
      if (!qty) break
      const q = Math.min(qty, o.qty)
      trades.push([o.id, o.price, q])
      o.qty -= q
      qty -= q
    }
    this.orders = this.orders.filter((o) => o.qty > 0)
    if (qty && !market) this.orders.push({ id, side, price, qty, seq: this.seq++ })
    return trades
  }

  cancel(id) {
    this.orders = this.orders.filter((o) => o.id !== id)
  }

  depth(side) {
    const by = new Map()
    for (const o of this.orders) if (o.side === side) by.set(o.price, (by.get(o.price) ?? 0) + o.qty)
    return [...by].sort((a, b) => (side === BUY ? b[0] - a[0] : a[0] - b[0]))
  }

  ahead(id) {
    const me = this.orders.find((o) => o.id === id)
    return this.orders
      .filter((o) => o.side === me.side && o.price === me.price && o.seq < me.seq)
      .reduce((a, o) => a + o.qty, 0)
  }
}

test("book matches a naive matcher on random orders", () => {
  const rand = mulberry32(11)
  for (let run = 0; run < 20; run++) {
    const book = new Book()
    const naive = new Naive()
    const live = []
    for (let id = 1; id <= 400; id++) {
      const u = rand()
      if (u < 0.2 && live.length) {
        const victim = live.splice(Math.floor(rand() * live.length), 1)[0]
        book.cancel(victim)
        naive.cancel(victim)
      } else {
        const side = rand() < 0.5 ? BUY : SELL
        const market = u > 0.92
        const postOnly = !market && rand() < 0.2
        const px = 95 + Math.floor(rand() * 11)
        const qty = 1 + Math.floor(rand() * 4)
        const events = book.submit(id, side, px, qty, { market, postOnly })
        const trades = events.filter((e) => e.type === "trade").map((e) => [e.maker, e.price, e.qty])
        assert.deepEqual(trades, naive.submit(id, side, px, qty, market, postOnly))
        if (book.orders.has(id)) live.push(id)
      }
      assert.equal(book.audit(), "")
      assert.deepEqual(depth(book, BUY), naive.depth(BUY))
      assert.deepEqual(depth(book, SELL), naive.depth(SELL))
    }
    for (const id of book.orders.keys()) assert.equal(book.ahead(id), naive.ahead(id))
  }
})

test("post only never trades and bad orders are rejected", () => {
  const b = new Book()
  b.submit(1, SELL, 101, 2)
  const crossing = b.submit(2, BUY, 101, 1, { postOnly: true })
  assert.deepEqual(
    crossing.map((e) => [e.type, e.reason]),
    [
      ["accepted", undefined],
      ["canceled", "postOnly"],
    ],
  )
  assert.equal(b.submit(1, BUY, 99, 1)[0].reason, "duplicate")
  assert.equal(b.submit(3, BUY, 99, 0)[0].reason, "qty")
  assert.equal(b.submit(4, BUY, -1, 1)[0].reason, "price")
  assert.equal(b.cancel(42)[0].reason, "unknown")
  const sweep = b.submit(5, BUY, 0, 5, { market: true })
  assert.deepEqual(
    sweep.slice(1).map((e) => [e.type, e.qty]),
    [
      ["trade", 2],
      ["canceled", 3],
    ],
  )
  assert.equal(b.ask, null)
})

test("seeded book matches python", () => {
  const b = new Book()
  let n = 1
  new ZeroIntelligence(skyPerfect()).seed(b, 10_000, 10_001, () => n++)
  assert.deepEqual(depth(b, BUY), ref.seeded.buy)
  assert.deepEqual(depth(b, SELL), ref.seeded.sell)
})

test("glft quotes match python", () => {
  for (const c of ref.models) {
    const m = new Model(c.params)
    const { bids, asks } = m.quotes()
    const a = m.approx()
    c.bids.forEach((x, i) => (x === null ? assert.ok(Number.isNaN(bids[i])) : close(bids[i], x, 1e-9, "bid")))
    c.asks.forEach((x, i) => (x === null ? assert.ok(Number.isNaN(asks[i])) : close(asks[i], x, 1e-9, "ask")))
    c.approxBids.forEach((x, i) => close(a.bids[i], x, 1e-12, "approx bid"))
    c.approxAsks.forEach((x, i) => close(a.asks[i], x, 1e-12, "approx ask"))
  }
})

const agree = (xs, want, what, n = xs.length, sdTol = 0.1) => {
  close(mean(xs), want.mean, 4 * want.sd * Math.sqrt(1 / n + 1 / ref.n), `${what} mean`)
  close(sd(xs), want.sd, sdTol * want.sd, `${what} sd`)
}

test("zero intelligence flow matches python", () => {
  const dmid = []
  const spread = []
  const top = []
  for (let s = 0; s < 1000; s++) {
    const m = new Market(new ZeroIntelligence(skyPerfect()), [], mulberry32(5000 + s))
    m.seed(10_000, 10_001)
    m.warmUp(30)
    const start = m.mark()
    const sp = []
    const dp = []
    for (let j = 0; j < 60; j++) {
      m.warmUp(1)
      const { bid, ask } = m.book
      if (bid !== null && ask !== null) {
        sp.push(ask - bid)
        dp.push(m.book.level(BUY, bid).qty)
      }
    }
    dmid.push(m.mark() - start)
    spread.push(mean(sp))
    top.push(mean(dp))
  }
  agree(dmid, ref.flow.dmid, "mid change")
  agree(spread, ref.flow.spread, "spread")
  agree(top, ref.flow.depth, "depth at the bid")
})

const BOTS = {
  glft: () => new GLFT({ sigma: 1, A: 2.1, k: 3.6, gamma: 0.08, Q: 5 }, { keep: 1 }),
  as: () => new AvellanedaStoikov({ gamma: 0.003, sigma: 1, k: 3.6, maxPosition: 5 }),
}

test("bot sessions match python", () => {
  const n = 400
  for (const [game, want] of Object.entries(ref.games)) {
    const makers = Object.fromEntries(Object.keys(want).map((k) => [k, BOTS[k]]))
    const rows = []
    for (let s = 0; s < n; s++) rows.push(session(new ZeroIntelligence(skyPerfect()), makers, mulberry32(9000 + s)).results)
    for (const [bot, stats] of Object.entries(want)) {
      for (const stat of ["pnl", "fills", "position"]) {
        agree(rows.map((r) => r[bot][stat]), stats[stat], `${game} ${bot} ${stat}`, n, 0.15)
      }
    }
  }
})

test("accounts and live orders stay consistent with the book", () => {
  const { market, results } = session(new ZeroIntelligence(skyPerfect()), BOTS, mulberry32(3))
  assert.equal(market.book.audit(), "")
  for (const { ctx } of Object.values(results)) {
    let q = 0
    let cash = 0
    for (const f of ctx.fills) {
      const sign = f.side === BUY ? 1 : -1
      q += sign * f.qty
      cash -= sign * f.price * f.qty
      assert.ok(f.maker && f.ahead >= 0 && f.left >= 0)
    }
    assert.equal(ctx.position, q)
    assert.equal(ctx.cash, cash)
    assert.ok(Math.abs(ctx.position) <= 5)
    const mine = [...market.book.orders.values()].filter((o) => o.owner === ctx.owner)
    assert.deepEqual(mine.map((o) => o.id).sort(), [...ctx.orders.keys()].sort())
    assert.ok(mine.length <= 2)
  }
})
