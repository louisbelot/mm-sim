import { PAPER, fillProb, quotes, spread, world } from "./as.js"

export class Maker {
  constructor() {
    this.q = 0
    this.x = 0
    this.trades = []
    this.qPath = [0]
    this.bids = []
    this.asks = []
  }

  trade(j, side, px, mid, who) {
    this.q += side
    this.x -= side * px
    this.trades.push({ j, side, px, mid, who })
  }

  wealth(s) {
    return this.x + this.q * s
  }
}

export const asPolicy = (p) => (j, s, q) =>
  quotes(s, q, p.T - j * p.dt, p.gamma, p.sigma, p.k)

export function symmetricPolicy(p) {
  const half = spread(p.T / 2, p.gamma, p.sigma, p.k) / 2
  return (j, s) => [s - half, s + half]
}

export function fill(w, j, m, bid, ask) {
  const s = w.mid[j]
  m.bids.push(bid)
  m.asks.push(ask)
  if (w.uBid[j] < fillProb(s - bid, w.p)) m.trade(j, 1, bid, s, "flow")
  if (w.uAsk[j] < fillProb(ask - s, w.p)) m.trade(j, -1, ask, s, "flow")
}

export function run(w, policy) {
  const m = new Maker()
  for (let j = 0; j < w.n; j++) {
    const [bid, ask] = policy(j, w.mid[j], m.q)
    fill(w, j, m, bid, ask)
    m.qPath.push(m.q)
  }
  return m
}

export function split(m, mid) {
  let edge = 0
  for (const t of m.trades) edge += t.side * (t.mid - t.px)
  let inventory = 0
  for (let j = 1; j < m.qPath.length; j++) inventory += m.qPath[j] * (mid[j] - mid[j - 1])
  return { edge, inventory }
}

export function distribution(sessions, p = PAPER, seed = 1) {
  const out = new Float64Array(sessions)
  for (let i = 0; i < sessions; i++) {
    const w = world(seed + i, p)
    out[i] = run(w, asPolicy(p)).wealth(w.mid[w.n])
  }
  return out
}

export class Session {
  constructor(w, mode) {
    this.w = w
    this.mode = mode
    this.j = 0
    this.bot = new Maker()
    this.you = new Maker()
    this.policy = asPolicy(w.p)
    this.half = spread(w.p.T, w.p.gamma, w.p.sigma, w.p.k) / 2
    this.lean = 0
  }

  get s() {
    return this.w.mid[this.j]
  }

  get done() {
    return this.j >= this.w.n
  }

  botQuotes() {
    return this.policy(this.j, this.s, this.bot.q)
  }

  yourQuotes() {
    return [this.s - this.half - this.lean, this.s + this.half - this.lean]
  }

  take(side) {
    if (this.mode !== "take" || this.done) return null
    const [bid, ask] = this.botQuotes()
    const px = side > 0 ? ask : bid
    this.you.trade(this.j, side, px, this.s, "you")
    this.bot.trade(this.j, -side, px, this.s, "you")
    return px
  }

  advance() {
    if (this.done) return
    const [bid, ask] = this.botQuotes()
    fill(this.w, this.j, this.bot, bid, ask)
    if (this.mode === "make") {
      const [b, a] = this.yourQuotes()
      fill(this.w, this.j, this.you, b, a)
    }
    this.j++
    this.bot.qPath.push(this.bot.q)
    this.you.qPath.push(this.you.q)
  }

  alone() {
    return run(this.w, this.policy)
  }
}

export function marks(m, mid, upto) {
  const out = new Float64Array(upto + 1)
  let q = 0
  let x = 0
  let i = 0
  for (let j = 0; j <= upto; j++) {
    while (i < m.trades.length && m.trades[i].j < j) {
      q += m.trades[i].side
      x -= m.trades[i].side * m.trades[i].px
      i++
    }
    out[j] = x + q * mid[j]
  }
  return out
}
