import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { test } from "node:test"

import { BUY, Book, SELL } from "../js/book.js"
import { HawkesFlow, ZeroIntelligence, skyPerfect, symmetricHawkes } from "../js/cst.js"
import { Fundamental, InformedTrader } from "../js/informed.js"
import { Live, byLevel, edgeOf, makers, midAt, split, world } from "../js/live.js"
import { Market, session } from "../js/market.js"
import { mulberry32 } from "../js/rng.js"
import { mean, sd } from "../js/stats.js"

const ref = JSON.parse(readFileSync(new URL("./queue_reference.json", import.meta.url)))

const close = (got, want, tol, what) =>
  assert.ok(Math.abs(got - want) <= tol, `${what}: got ${got}, want ${want} +- ${tol}`)

function kurtosis(xs) {
  const m = mean(xs)
  const v = sd(xs) ** 2
  return xs.reduce((a, x) => a + (x - m) ** 4, 0) / xs.length / v ** 2
}

const agree = (xs, want, what) => {
  const both = 1 / xs.length + 1 / ref.n
  close(mean(xs), want.mean, 4 * want.sd * Math.sqrt(both), `${what} mean`)
  close(sd(xs), want.sd, 4 * want.sd * Math.sqrt(((kurtosis(xs) - 1) / 4) * both), `${what} sd`)
}

const SKY = skyPerfect()
const HAWKES = symmetricHawkes(SKY.mu, 0.6, 0.1, 6)

class Clock {
  constructor(every) {
    this.every = every
    this.times = []
  }

  onStart(ctx) {
    ctx.setTimer(this.every)
  }

  onTimer(ctx) {
    this.times.push(ctx.t)
    ctx.setTimer(this.every)
  }

  onBook() {}

  onFill() {}
}

test("timers fire on time between flow events", () => {
  const clock = new Clock(0.25)
  const m = new Market(new ZeroIntelligence(SKY), [clock], mulberry32(4))
  m.seed(10_000, 10_001)
  m.warmUp(30)
  m.start(10)
  for (let j = 1; j <= 100; j++) m.loop(30 + j * 0.1)
  assert.equal(clock.times.length, 40)
  clock.times.forEach((t, k) => close(t, 30 + 0.25 * (k + 1), 1e-9, `timer ${k}`))
  for (let i = 1; i < m.tape.length; i++) assert.ok(m.tape[i][0] >= m.tape[i - 1][0])
})

test("hawkes flow keeps the mean market order rate on a deep book", () => {
  const deep = { ...SKY, lam: SKY.lam.map((x) => 3 * x) }
  const f = new HawkesFlow(deep, HAWKES)
  const m = new Market(f, [], mulberry32(11))
  m.seed(10_000, 10_001)
  m.warmUp(30)
  const n0 = f.markets.length
  const w = 20_000
  m.warmUp(w)
  const rate = 2 * SKY.mu
  close((f.markets.length - n0) / w, rate, (4 * Math.sqrt(rate / w)) / (1 - 0.7), "market orders a minute")
})

function rescaled(markets, h, side) {
  const out = []
  let s = [0, 0]
  let last = markets[0][0]
  let big = 0
  for (const [t, k] of markets) {
    const dt = t - last
    for (let j = 0; j < 2; j++) {
      const b = h.beta[side][j]
      big += (s[j] * (1 - Math.exp(-b * dt))) / b
      s[j] *= Math.exp(-b * dt)
    }
    big += h.mu[side] * dt
    if (k === side) out.push(big)
    s[k] += h.alpha[side][k]
    last = t
  }
  return out.slice(1).map((x, i) => x - out[i])
}

function ksExp(gaps) {
  const xs = [...gaps].sort((a, b) => a - b)
  let d = 0
  xs.forEach((x, i) => {
    const F = 1 - Math.exp(-x)
    d = Math.max(d, Math.abs(F - i / xs.length), Math.abs(F - (i + 1) / xs.length))
  })
  return d
}

test("rescaled market order gaps are unit exponential under the true model only", () => {
  const f = new HawkesFlow(SKY, HAWKES)
  const m = new Market(f, [], mulberry32(12))
  m.seed(10_000, 10_001)
  m.warmUp(2000)
  for (const side of [BUY, SELL]) {
    const gaps = rescaled(f.markets, HAWKES, side)
    const crit = 1.63 / Math.sqrt(gaps.length)
    assert.ok(ksExp(gaps) < crit, `true model, side ${side}: D ${ksExp(gaps)}`)
    const poisson = { mu: [SKY.mu, SKY.mu], alpha: [[0, 0], [0, 0]], beta: HAWKES.beta }
    assert.ok(ksExp(rescaled(f.markets, poisson, side)) > 3 * crit, "poisson rates")
  }
})

test("hawkes flow alone matches python", () => {
  const dmid = []
  const markets = []
  for (let s = 0; s < 600; s++) {
    const f = new HawkesFlow(SKY, HAWKES)
    const m = new Market(f, [], mulberry32(7000 + s))
    m.seed(10_000, 10_001)
    m.warmUp(30)
    const start = m.mark()
    const n0 = f.markets.length
    m.warmUp(60)
    dmid.push(m.mark() - start)
    markets.push(f.markets.length - n0)
  }
  agree(dmid, ref.hawkes.dmid, "mid change")
  agree(markets, ref.hawkes.markets, "market orders")
})

function fakeCtx(bid, ask) {
  const book = new Book()
  book.submit(1, BUY, bid, 3)
  book.submit(2, SELL, ask, 3)
  const sent = []
  return { t: 1, book, sent, submit: (side, px, qty, o) => sent.push([side, qty, o.market]) }
}

test("informed trader takes only when the value is outside the quotes", () => {
  const cases = [
    [103, 0, [[BUY, 1, true]]],
    [97, 0, [[SELL, 1, true]]],
    [100, 0, []],
    [101, 0, []],
    [103, 2, []],
    [103.5, 2, [[BUY, 1, true]]],
  ]
  for (const [v, edge, want] of cases) {
    const value = new Fundamental(0, mulberry32(1))
    value.start(0, v)
    const ctx = fakeCtx(99, 101)
    new InformedTrader(value, 1, mulberry32(2), { edge }).trade(ctx)
    assert.deepEqual(ctx.sent, want, `v ${v} edge ${edge}`)
  }
})

test("fundamental increments have variance sigma squared dt", () => {
  const value = new Fundamental(1.5, mulberry32(9))
  value.start(0, 0)
  const dx = []
  let x = 0
  for (let i = 1; i <= 40_000; i++) {
    const y = value.at(i * 0.1)
    dx.push(y - x)
    x = y
  }
  close(mean(dx), 0, 4 * Math.sqrt(0.225 / 40_000), "mean")
  close(sd(dx) ** 2, 0.225, 0.225 * 0.03, "variance")
})

test("frame by frame equals one shot with hawkes flow and an informed trader", () => {
  const cfg = { glft: 0.08, as: 0.003, hawkes: true, informed: true }
  for (const seed of [3, 41]) {
    const live = new Live(seed, cfg)
    while (!live.done) live.step()
    const { flow, crowd } = world(cfg, seed)
    const { results } = session(flow, makers(cfg), mulberry32(seed), { crowd })
    for (const name of live.names) {
      const ctx = live.ctx[name]
      assert.deepEqual(ctx.fills, results[name].ctx.fills)
      const rows = byLevel(ctx, live.market.tape)
      assert.equal(rows[0].n + rows[1].n, ctx.fills.length)
    }
    const trader = live.informed
    assert.ok(trader.agent.value.path.length > 30)
    assert.ok(trader.fills.length > 0 && trader.fills.every((f) => !f.maker))
    assert.equal(live.value.length, live.mid.length)
  }
})

test("a fill on a one-sided book has no edge, and the split still adds up", () => {
  const cfg = { glft: 0.08, as: 0.003, hawkes: true, informed: false }
  let oneSided = 0
  for (let seed = 1; seed <= 40; seed++) {
    const { flow, crowd } = world(cfg, seed)
    const { market, results } = session(flow, makers(cfg), mulberry32(seed), { crowd })
    const c = market.mark()
    for (const { ctx } of Object.values(results)) {
      for (const f of ctx.fills) {
        if (f.mid === null) oneSided++
        else assert.ok(edgeOf(f) > 0, "a post-only fill is on the good side of the mid")
      }
      const { edge, inventory, pnl } = split(ctx, c)
      assert.ok(edge >= 0 && Math.abs(edge + inventory - pnl) < 1e-9)
    }
  }
  assert.ok(oneSided > 0)
})

const STATS = ["pnl", "fills", "position", "front", "markout"]

function stats(ctx, tape, close) {
  let markout = 0
  for (const f of ctx.fills) markout += (f.side === BUY ? 1 : -1) * f.qty * (midAt(tape, f.t + 1) - f.price)
  const front = ctx.fills.filter((f) => f.ahead === 0).length
  return { pnl: ctx.value(close), fills: ctx.fills.length, position: ctx.position, front, markout }
}

test("sessions with hawkes flow and informed trading match python", () => {
  const n = 250
  for (const [key, want] of Object.entries(ref.games)) {
    const [w, game] = key.split("/")
    const cfg = {
      glft: 0.08,
      as: game === "both" ? 0.003 : null,
      hawkes: w.startsWith("hawkes"),
      informed: w.endsWith("informed"),
    }
    const rows = []
    for (let s = 0; s < n; s++) {
      const seed = 20_000 + s
      const { flow, crowd } = world(cfg, seed)
      const { market, results } = session(flow, makers(cfg), mulberry32(seed), { crowd })
      const c = market.mark()
      rows.push(Object.fromEntries(Object.entries(results).map(([k, r]) => [k, stats(r.ctx, market.tape, c)])))
    }
    for (const [bot, st] of Object.entries(want)) {
      for (const stat of STATS) agree(rows.map((r) => r[bot][stat]), st[stat], `${key} ${bot} ${stat}`)
    }
  }
})
