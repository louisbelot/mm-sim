import assert from "node:assert/strict"
import { test } from "node:test"

import { BUY } from "../js/book.js"
import { ZeroIntelligence, skyPerfect } from "../js/cst.js"
import { DT, Live, MINUTES, byQueue, makers, midAt, split } from "../js/live.js"
import { session } from "../js/market.js"
import { mulberry32 } from "../js/rng.js"

const CFG = { glft: 0.08, as: 0.003 }

test("stepping frame by frame replays the one-shot session", () => {
  for (const seed of [1, 2, 77]) {
    const live = new Live(seed, CFG)
    while (!live.done) live.step()
    const { results } = session(new ZeroIntelligence(skyPerfect()), makers(CFG), mulberry32(seed))
    for (const name of live.names) {
      const a = live.ctx[name]
      const b = results[name].ctx
      assert.deepEqual(a.fills, b.fills)
      assert.equal(a.position, b.position)
      assert.equal(a.cash, b.cash)
    }
    assert.equal(live.mid.length, MINUTES / DT + 1)
    assert.equal(live.minute, MINUTES)
  }
})

test("edge plus inventory is the pnl", () => {
  const live = new Live(5, CFG)
  while (!live.done) live.step()
  const close = live.market.mark()
  for (const name of live.names) {
    const ctx = live.ctx[name]
    const { edge, inventory, pnl } = split(ctx, close)
    assert.ok(Math.abs(edge + inventory - pnl) < 1e-9)
    assert.equal(pnl, ctx.value(close))
    for (const f of ctx.fills) {
      const sign = f.side === BUY ? 1 : -1
      assert.ok(sign * (f.mid - f.price) > 0, "a post-only fill is on the good side of the mid")
    }
  }
})

test("queue table counts every fill once", () => {
  const live = new Live(9, CFG)
  while (!live.done) live.step()
  for (const name of live.names) {
    const ctx = live.ctx[name]
    const rows = byQueue(ctx, live.market.tape)
    assert.equal(
      rows.reduce((a, r) => a + r.n, 0),
      ctx.fills.length,
    )
    assert.equal(rows[0].n, ctx.fills.filter((f) => f.ahead === 0).length)
  }
})

test("mid lookup is a step function of the tape", () => {
  const tape = [
    [0, 10],
    [1, 10.5],
    [2.5, 11],
  ]
  assert.equal(midAt(tape, 0), 10)
  assert.equal(midAt(tape, 0.99), 10)
  assert.equal(midAt(tape, 1), 10.5)
  assert.equal(midAt(tape, 9), 11)
})

test("a single bot can run alone", () => {
  const live = new Live(3, { glft: null, as: 0.003 })
  assert.deepEqual(live.names, ["as"])
  while (!live.done) live.step()
  assert.equal(live.market.book.audit(), "")
})
