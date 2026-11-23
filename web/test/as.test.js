import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { test } from "node:test"

import { PAPER, fillProb, quotes, reservationPrice, spread, steps, world } from "../js/as.js"
import {
  Session,
  asPolicy,
  distribution,
  marks,
  run,
  split,
  symmetricPolicy,
} from "../js/session.js"
import { histogram, mean, rank, sd } from "../js/stats.js"

const ref = JSON.parse(readFileSync(new URL("./reference.json", import.meta.url)))

const close = (got, want, tol, what) =>
  assert.ok(Math.abs(got - want) <= tol, `${what}: got ${got}, want ${want} +- ${tol}`)

test("quotes match the python model", () => {
  for (const c of ref.quotes) {
    const [bid, ask] = quotes(c.s, c.q, c.tau, c.gamma, c.sigma, c.k)
    close(reservationPrice(c.s, c.q, c.tau, c.gamma, c.sigma), c.r, 1e-12, "r")
    close(spread(c.tau, c.gamma, c.sigma, c.k), c.spread, 1e-12, "spread")
    close(bid, c.bid, 1e-12, "bid")
    close(ask, c.ask, 1e-12, "ask")
  }
})

test("paper tables agree with the python simulation", () => {
  const n = 4000
  for (const c of ref.sims) {
    const p = { ...PAPER, gamma: c.gamma }
    const policy = c.strategy === "inventory" ? asPolicy(p) : symmetricPolicy(p)
    const profit = []
    const q = []
    for (let i = 0; i < n; i++) {
      const w = world(1000 + i, p)
      const m = run(w, policy)
      profit.push(m.wealth(w.mid[w.n]))
      q.push(m.q)
    }
    const what = `${c.gamma} ${c.strategy}`
    close(mean(profit), c.profit, 4 * c.sdProfit * Math.sqrt(1 / n + 1 / c.n), `${what} profit`)
    close(mean(q), c.q, 4 * c.sdQ * Math.sqrt(1 / n + 1 / c.n), `${what} q`)
    close(sd(profit), c.sdProfit, 0.08 * c.sdProfit, `${what} sd profit`)
    close(sd(q), c.sdQ, 0.08 * c.sdQ, `${what} sd q`)
  }
})

test("world is a seeded binomial walk", () => {
  const a = world(7)
  const b = world(7)
  assert.equal(a.n, steps(PAPER))
  assert.deepEqual(a.mid, b.mid)
  assert.notDeepEqual(world(8).mid, a.mid)
  const dx = PAPER.sigma * Math.sqrt(PAPER.dt)
  for (let j = 0; j < a.n; j++) close(Math.abs(a.mid[j + 1] - a.mid[j]), dx, 1e-9, "step")
})

test("fill probability is capped at one", () => {
  assert.equal(fillProb(-10, PAPER), 1)
  close(fillProb(1, PAPER), 140 * Math.exp(-1.5) * 0.005, 1e-15, "intensity")
})

test("wealth splits exactly into edge and inventory", () => {
  for (const mode of ["take", "make"]) {
    const s = new Session(world(3), mode)
    let k = 0
    while (!s.done) {
      if (mode === "take" && k % 7 === 0) s.take(k % 2 ? 1 : -1)
      if (mode === "make") s.lean = Math.sin(k / 9)
      s.advance()
      k++
    }
    const mid = s.w.mid
    for (const m of [s.bot, s.you]) {
      const { edge, inventory } = split(m, mid)
      close(edge + inventory, m.wealth(mid[s.w.n]), 1e-9, mode)
    }
  }
})

test("marked path ends at final wealth", () => {
  const w = world(9)
  const m = run(w, asPolicy(PAPER))
  const path = marks(m, w.mid, w.n)
  assert.equal(path[0], 0)
  close(path[w.n], m.wealth(w.mid[w.n]), 1e-9, "end")
  const half = marks(m, w.mid, 100)
  assert.deepEqual(half, path.slice(0, 101))
})

test("quoting like the bot earns exactly what the bot earns", () => {
  const p = PAPER
  const s = new Session(world(11), "make")
  while (!s.done) {
    const tau = p.T - s.j * p.dt
    s.half = spread(tau, p.gamma, p.sigma, p.k) / 2
    s.lean = s.you.q * p.gamma * p.sigma ** 2 * tau
    s.advance()
  }
  assert.deepEqual(s.you.trades, s.bot.trades)
  assert.equal(s.you.wealth(100), s.bot.wealth(100))
})

test("without your trades the bot is the bot alone", () => {
  const s = new Session(world(5), "take")
  while (!s.done) s.advance()
  const alone = s.alone()
  assert.deepEqual(s.bot.trades, alone.trades)
  assert.deepEqual(s.bot.qPath, alone.qPath)
})

function play(seed, rule) {
  const s = new Session(world(seed), "take")
  while (!s.done) {
    for (let i = 0; i < 5; i++) {
      const side = rule(s)
      if (!side) break
      s.take(side)
    }
    s.advance()
  }
  return s
}

const reader = (s) => {
  const [bid, ask] = s.botQuotes()
  return ask < s.s ? 1 : bid > s.s ? -1 : 0
}

const insider = (s) => {
  const [bid, ask] = s.botQuotes()
  return ask < s.w.tip - 0.5 ? 1 : bid > s.w.tip + 0.5 ? -1 : 0
}

test("trading against the skew has positive edge on every trade", () => {
  let n = 0
  for (let seed = 1; seed <= 400; seed++) {
    const s = play(seed, reader)
    for (const t of s.you.trades) assert.ok(t.side * (t.mid - t.px) > 0)
    n += s.you.trades.length
  }
  assert.ok(n > 20)
})

test("the bot passes an insider's information on to the flow", () => {
  const you = []
  const gain = []
  for (let seed = 1; seed <= 300; seed++) {
    const s = play(seed, insider)
    const end = s.w.mid[s.w.n]
    you.push(s.you.wealth(end))
    gain.push(s.bot.wealth(end) - s.alone().wealth(end))
  }
  assert.ok(mean(you) > 5 * (sd(you) / Math.sqrt(you.length)))
  assert.ok(mean(gain) > 5 * (sd(gain) / Math.sqrt(gain.length)))
})

test("distribution and stats", () => {
  const d = distribution(500)
  close(mean(d), ref.sims[0].profit, 4 * ref.sims[0].sdProfit / Math.sqrt(500), "mean")
  assert.equal(rank([1, 2, 3, 4], 3), 0.5)
  const h = histogram([0, 0.5, 1.5, 9, -3], 0, 2, 2)
  assert.deepEqual(h, [3, 2])
})
