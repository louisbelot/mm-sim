import { BUY } from "./js/book.js"
import { ZeroIntelligence, skyPerfect } from "./js/cst.js"
import { makers, midAt, split } from "./js/live.js"
import { session } from "./js/market.js"
import { mulberry32 } from "./js/rng.js"
import { mean, sd } from "./js/stats.js"

const N = Number(process.argv[2] ?? 2000)
const GAMES = {
  "GLFT alone": { glft: 0.08, as: null },
  "AS alone": { glft: null, as: 0.003 },
  together: { glft: 0.08, as: 0.003 },
}
const se = (xs) => sd(xs) / Math.sqrt(xs.length)
const f2 = (x) => x.toFixed(2)

for (const [game, cfg] of Object.entries(GAMES)) {
  const out = {}
  for (let s = 0; s < N; s++) {
    const { market, results } = session(new ZeroIntelligence(skyPerfect()), makers(cfg), mulberry32(s + 1))
    const close = market.mark()
    for (const [name, r] of Object.entries(results)) {
      out[name] ??= { pnl: [], edge: [], after: [], buckets: [[], [], [], []] }
      const o = out[name]
      const { edge, inventory, pnl } = split(r.ctx, close)
      o.pnl.push(pnl)
      o.edge.push(edge)
      o.after.push(inventory)
      for (const f of r.ctx.fills) {
        const sign = f.side === BUY ? 1 : -1
        o.buckets[Math.min(f.ahead, 3)].push(sign * (midAt(market.tape, f.t + 1) - f.price))
      }
    }
  }
  for (const [name, o] of Object.entries(out)) {
    const fills = o.buckets.reduce((a, b) => a + b.length, 0)
    console.log(
      `${game} ${name}: pnl ${f2(mean(o.pnl))} (se ${f2(se(o.pnl))}), edge ${f2(mean(o.edge))}, after ${f2(mean(o.after))}, fills ${f2(fills / N)}`,
    )
    o.buckets.forEach((b, k) => {
      const share = (100 * b.length) / fills
      const m = b.length ? `${f2(mean(b))} (se ${f2(se(b))})` : "-"
      console.log(`  ahead ${k === 3 ? "3+" : k}: ${share.toFixed(1)}% of fills, 1-min markout ${m}`)
    })
  }
}
