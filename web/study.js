import { BUY } from "./js/book.js"
import { edgeOf, makers, midAt, split, world } from "./js/live.js"
import { session } from "./js/market.js"
import { mulberry32 } from "./js/rng.js"
import { mean, sd } from "./js/stats.js"

const N = Number(process.argv[2] ?? 2000)
const WORLDS = {
  poisson: { hawkes: false, informed: false },
  informed: { hawkes: false, informed: true },
  hawkes: { hawkes: true, informed: false },
  "hawkes+informed": { hawkes: true, informed: true },
}
const GAMES = {
  "GLFT alone": { glft: 0.08, as: null },
  together: { glft: 0.08, as: 0.003 },
}
const se = (xs) => sd(xs) / Math.sqrt(xs.length)
const f2 = (x) => x.toFixed(2)
const cell = (b) => (b.length ? `${f2(mean(b))} (${f2(se(b))})` : "-")
const pair = (b) => {
  if (!b.length) return "-"
  const move = b.map(([m, e]) => m - e)
  return `edge ${f2(mean(b.map((x) => x[1])))} move ${f2(mean(move))} (${f2(se(move))})`
}

for (const [w, flags] of Object.entries(WORLDS)) {
  for (const [game, bots] of Object.entries(GAMES)) {
    const cfg = { ...bots, ...flags }
    const out = {}
    for (let s = 0; s < N; s++) {
      const { flow, crowd } = world(cfg, s + 1)
      const { market, results } = session(flow, makers(cfg), mulberry32(s + 1), { crowd })
      const close = market.mark()
      for (const [name, r] of Object.entries(results)) {
        out[name] ??= { pnl: [], edge: [], after: [], buckets: [[], [], [], []], cleared: [[], []] }
        const o = out[name]
        const { edge, inventory, pnl } = split(r.ctx, close)
        o.pnl.push(pnl)
        o.edge.push(edge)
        o.after.push(inventory)
        for (const f of r.ctx.fills) {
          const x = [(f.side === BUY ? 1 : -1) * (midAt(market.tape, f.t + 1) - f.price), edgeOf(f)]
          o.buckets[Math.min(f.ahead, 3)].push(x)
          o.cleared[f.left === 0 ? 1 : 0].push(x)
        }
      }
    }
    for (const [name, o] of Object.entries(out)) {
      const fills = o.buckets.reduce((a, b) => a + b.length, 0)
      console.log(
        `${w} | ${game} | ${name}: pnl ${cell(o.pnl)}, edge ${f2(mean(o.edge))}, after ${f2(mean(o.after))}, fills ${f2(fills / N)}`,
      )
      const rows = o.buckets.map((b, k) => `${k === 3 ? "3+" : k} ${((100 * b.length) / fills).toFixed(1)}% ${pair(b)}`)
      console.log(`  ahead at entry, share, edge and 1-min move: ${rows.join(" | ")}`)
      const cleared = ((100 * o.cleared[1].length) / fills).toFixed(1)
      console.log(`  level left standing ${pair(o.cleared[0])}, level cleared ${pair(o.cleared[1])} (${cleared}%)`)
    }
  }
}
