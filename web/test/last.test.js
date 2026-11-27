import assert from "node:assert/strict"
import { test } from "node:test"

import { BUY, Book, SELL } from "../js/book.js"
import { Touch, others } from "../js/bots.js"
import { ZeroIntelligence, drawSize, itchSizes, skyPerfect, zetaSizes } from "../js/cst.js"
import { Live, makers, world } from "../js/live.js"
import { Market, session } from "../js/market.js"
import { mulberry32 } from "../js/rng.js"

function fakeCtx() {
  const book = new Book()
  book.submit(1, BUY, 99, 5)
  book.submit(2, SELL, 102, 5)
  const sent = []
  let n = 100
  const ctx = {
    book,
    owner: 9,
    bid: 99,
    ask: 102,
    mid: 100.5,
    position: 0,
    orders: new Map(),
    sent,
    submit: (side, px) => {
      sent.push([side, px])
      return n++
    },
    cancel: () => {},
  }
  return ctx
}

test("long zeta head is one over riemann zeta", () => {
  const s = zetaSizes(2.5, 1_000_000)
  assert.ok(Math.abs(s.pmf[0] - 1 / 1.341487257250917) < 1e-9)
  const itch = itchSizes()
  assert.equal(itch.pmf.length, 20)
  assert.ok(itch.mean > 1.4 && itch.mean < 1.6)
})

test("drawn sizes follow the pmf", () => {
  const s = zetaSizes(2, 8)
  const rand = mulberry32(3)
  const n = 200_000
  const counts = new Array(8).fill(0)
  for (let i = 0; i < n; i++) counts[drawSize(s, rand) - 1]++
  s.pmf.forEach((p, k) => assert.ok(Math.abs(counts[k] - n * p) < 4 * Math.sqrt(n * p * (1 - p)), `size ${k + 1}`))
})

test("random sizes keep the share rate on a deep book", () => {
  const sky = skyPerfect()
  const deep = { ...sky, lam: sky.lam.map((x) => 3 * x) }
  const sizes = itchSizes()
  const f = new ZeroIntelligence(deep, 0, sizes)
  const m = new Market(f, [], mulberry32(21))
  m.seed(10_000, 10_001)
  const shares = []
  let walks = 0
  const exec = m.execute.bind(m)
  m.execute = (sender, cmd) =>
    exec(sender, (b) => {
      const ev = cmd(b)
      if (ev[0].type === "accepted" && ev[0].price === 0) {
        shares.push(ev[0].qty)
        walks += new Set(ev.filter((e) => e.type === "trade").map((e) => e.price)).size > 1
      }
      return ev
    })
  const T = 3000
  m.warmUp(T)
  assert.equal(m.book.audit(), "")
  const rate = (2 * deep.mu * T) / sizes.mean
  assert.ok(Math.abs(shares.length - rate) < 4 * Math.sqrt(rate))
  const k2 = sizes.pmf.reduce((a, p, i) => a + p * (i + 1) ** 2, 0)
  const total = shares.reduce((a, b) => a + b, 0)
  assert.ok(Math.abs(total - 2 * deep.mu * T) < 4 * Math.sqrt(rate * k2))
  assert.ok(walks > 0)
})

for (const [thin, back, want] of [
  [1, 1, [99, 102]],
  [6, 1, [99, 102]],
  [6, 2, [98, 103]],
]) {
  test(`thin ${thin} back ${back} steps back from an empty price`, () => {
    const ctx = fakeCtx()
    new Touch({ thin, back }).requote(ctx)
    assert.deepEqual(ctx.sent, [
      [BUY, want[0]],
      [SELL, want[1]],
    ])
  })
}

test("own orders do not count as company", () => {
  const ctx = fakeCtx()
  ctx.book.submit(50, BUY, 100, 3, { owner: 9 })
  assert.equal(others(ctx.book, BUY, 100, 9), 0)
  assert.equal(others(ctx.book, BUY, 99, 9), 5)
  new Touch({ thin: 1 }).requote(ctx)
  assert.deepEqual(ctx.sent[0], [BUY, 99])
  const off = fakeCtx()
  new Touch().requote(off)
  assert.deepEqual(off.sent, [
    [BUY, 100],
    [SELL, 101],
  ])
})

test("frame by frame equals one shot with big orders and the last-share rule", () => {
  const cfg = { glft: 0.08, as: 0.003, hawkes: true, informed: true, sizes: true, thin: 1 }
  for (const seed of [5, 77]) {
    const live = new Live(seed, cfg)
    while (!live.done) live.step()
    const { flow, crowd } = world(cfg, seed)
    const { results } = session(flow, makers(cfg), mulberry32(seed), { crowd })
    for (const name of live.names) assert.deepEqual(live.ctx[name].fills, results[name].ctx.fills)
    assert.ok(live.ctx.glft.fills.length > 0)
  }
})
