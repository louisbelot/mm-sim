import { BUY, SELL } from "./book.js"
import { AvellanedaStoikov, GLFT } from "./bots.js"
import { bars, color, lines } from "./chart.js"
import { ZeroIntelligence, skyPerfect } from "./cst.js"
import { open, session } from "./market.js"
import { mulberry32 } from "./rng.js"
import { histogram, mean, rank, sd } from "./stats.js"

export const MINUTES = 60
export const WARM = 30
export const DT = 0.05
export const SIGMA = 1
export const A = 2.1
export const K = 3.6
export const Q = 5

export const MAKERS = {
  glft: (gamma) => () => new GLFT({ sigma: SIGMA, A, k: K, gamma, Q }, { keep: 1 }),
  as: (gamma) => () => new AvellanedaStoikov({ gamma, sigma: SIGMA, k: K, maxPosition: Q }),
}

export const NAMES = { glft: "GLFT", as: "AS" }

export function makers(cfg) {
  const out = {}
  for (const name of Object.keys(MAKERS)) {
    if (cfg[name] !== null) out[name] = MAKERS[name](cfg[name])
  }
  return out
}

export function midAt(tape, t) {
  let lo = 0
  let hi = tape.length - 1
  while (lo < hi) {
    const m = (lo + hi + 1) >> 1
    if (tape[m][0] <= t) lo = m
    else hi = m - 1
  }
  return tape[lo][1]
}

export function split(ctx, close) {
  let edge = 0
  for (const f of ctx.fills) edge += (f.side === BUY ? 1 : -1) * f.qty * (f.mid - f.price)
  const pnl = ctx.value(close)
  return { edge, inventory: pnl - edge, pnl }
}

export function byQueue(ctx, tape, horizon = 1) {
  const rows = [0, 1, 2, 3].map((k) => ({ k, n: 0, markout: 0 }))
  for (const f of ctx.fills) {
    const r = rows[Math.min(f.ahead, 3)]
    r.n++
    r.markout += (f.side === BUY ? 1 : -1) * (midAt(tape, f.t + horizon) - f.price)
  }
  for (const r of rows) r.markout = r.n ? r.markout / r.n : NaN
  return rows
}

export class Live {
  constructor(seed, cfg) {
    this.names = Object.keys(makers(cfg))
    const { market, ctx } = open(new ZeroIntelligence(skyPerfect()), makers(cfg), mulberry32(seed), {
      minutes: MINUTES,
      warm: WARM,
    })
    this.market = market
    this.ctx = ctx
    this.start = market.t
    this.n = Math.round(MINUTES / DT)
    this.j = 0
    this.mid = []
    this.quotes = Object.fromEntries(this.names.map((n) => [n, { bid: [], ask: [] }]))
    this.q = Object.fromEntries(this.names.map((n) => [n, []]))
    this.pnl = Object.fromEntries(this.names.map((n) => [n, []]))
    this.sample()
  }

  get done() {
    return this.j >= this.n
  }

  get minute() {
    return this.market.t - this.start
  }

  step() {
    this.j++
    this.market.loop(this.done ? this.market.end : this.start + this.j * DT)
    this.sample()
  }

  sample() {
    const mark = this.market.mark()
    this.mid.push(mark)
    for (const name of this.names) {
      const ctx = this.ctx[name]
      const [bid, ask] = this.resting(name)
      this.quotes[name].bid.push(bid ? bid.price : NaN)
      this.quotes[name].ask.push(ask ? ask.price : NaN)
      this.q[name].push(ctx.position)
      this.pnl[name].push(ctx.value(mark))
    }
  }

  resting(name) {
    const ctx = this.ctx[name]
    return [BUY, SELL].map((side) => {
      for (const [id, o] of ctx.orders) {
        if (o.side === side && this.market.book.orders.has(id)) {
          return { id, price: o.price, ahead: this.market.book.ahead(id) }
        }
      }
      return null
    })
  }
}

export function distribution(cfg, n, first = 1_000_000) {
  const out = Object.fromEntries(Object.keys(makers(cfg)).map((k) => [k, []]))
  for (let s = 0; s < n; s++) {
    const { results } = session(new ZeroIntelligence(skyPerfect()), makers(cfg), mulberry32(first + s))
    for (const k in out) out[k].push(results[k].pnl)
  }
  return out
}

function page() {
  const $ = (id) => document.getElementById(id)
  const fmt = (v, d = 1) => (Number.isFinite(v) ? v.toFixed(d) : "")
  const signed = (v, d = 1) => (Number.isFinite(v) ? (v >= 0 ? "+" : "") + v.toFixed(d) : "")
  const SPEEDS = { slow: 0.25, normal: 1, fast: 4 }
  const GAMMAS = { glft: [0.02, 0.08, 0.3], as: [0.001, 0.003, 0.01] }
  const params = new URLSearchParams(location.search)
  const pick = (name, dflt) => {
    const v = params.get(name)
    if (v === "off") return null
    return GAMMAS[name].includes(Number(v)) ? Number(v) : dflt
  }
  const state = {
    seed: Number(params.get("seed")) || 1 + Math.floor(Math.random() * 99999),
    cfg: { glft: pick("glft", 0.08), as: pick("as", 0.003) },
  }
  const swatch = (name) => color(name === "as" ? "avs" : name)
  let s = null
  let running = false
  let last = 0
  let acc = 0
  const others = new Map()

  for (const name of Object.keys(GAMMAS)) {
    $(`${name}-gamma`).innerHTML =
      `<option value="off">off</option>` + GAMMAS[name].map((g) => `<option>${g}</option>`).join("")
  }

  function reset() {
    running = false
    acc = 0
    if (state.cfg.glft === null && state.cfg.as === null) state.cfg.glft = 0.08
    s = new Live(state.seed, state.cfg)
    $("seed").value = state.seed
    for (const name of Object.keys(GAMMAS)) $(`${name}-gamma`).value = String(state.cfg[name] ?? "off")
    $("results").hidden = true
    const q = new URLSearchParams({ seed: state.seed })
    for (const name of Object.keys(GAMMAS)) q.set(name, String(state.cfg[name] ?? "off"))
    history.replaceState(null, "", `?${q}`)
    render()
  }

  function tick(now) {
    if (!running) return
    acc += ((now - last) / 1000) * SPEEDS[$("speed").value]
    last = now
    while (acc >= DT && !s.done) {
      s.step()
      acc -= DT
    }
    render()
    if (s.done) finish()
    else requestAnimationFrame(tick)
  }

  function toggle() {
    if (s.done) reset()
    running = !running
    render()
    if (running) {
      last = performance.now()
      requestAnimationFrame(tick)
    }
  }

  function skip() {
    running = false
    while (!s.done) s.step()
    finish()
  }

  function ladder() {
    const canvas = $("depth")
    const dpr = window.devicePixelRatio || 1
    const { width, height } = canvas.getBoundingClientRect()
    canvas.width = Math.round(width * dpr)
    canvas.height = Math.round(height * dpr)
    const g = canvas.getContext("2d")
    g.setTransform(dpr, 0, 0, dpr, 0, 0)
    g.clearRect(0, 0, width, height)
    const book = s.market.book
    const owners = Object.fromEntries(s.names.map((n) => [s.ctx[n].owner, swatch(n)]))
    const center = Math.round(s.market.mark())
    const rows = 17
    const top = center + (rows >> 1)
    const h = height / rows
    const x0 = 54
    const unit = Math.min(14, (width - x0 - 40) / 12)
    g.font = "12px ui-sans-serif, system-ui, sans-serif"
    g.textBaseline = "middle"
    for (let r = 0; r < rows; r++) {
      const px = top - r
      const y = r * h
      const ask = book.level(SELL, px)
      const bid = book.level(BUY, px)
      const lvl = ask ?? bid
      if (r % 2) {
        g.fillStyle = color("panel")
        g.fillRect(0, y, width, h)
      }
      g.fillStyle = ask ? color("ask") : bid ? color("bid") : color("muted")
      g.textAlign = "right"
      g.fillText(String(px), x0 - 10, y + h / 2)
      if (!lvl) continue
      let x = x0
      for (const o of lvl.orders) {
        g.fillStyle = owners[o.owner] ?? color("flow")
        g.fillRect(x, y + 3, unit * o.qty - 2, h - 6)
        x += unit * o.qty
      }
      g.fillStyle = color("muted")
      g.textAlign = "left"
      g.fillText(String(lvl.qty), x + 4, y + h / 2)
    }
  }

  function positions() {
    const mark = s.market.mark()
    $("book").innerHTML = s.names
      .map((name) => {
        const ctx = s.ctx[name]
        const [bid, ask] = s.resting(name)
        const q = (o) => (o ? `${o.price} (${o.ahead})` : "")
        return `<tr class="${name}"><td>${NAMES[name]}</td><td>${ctx.position}</td><td>${q(bid)}</td><td>${q(ask)}</td><td>${fmt(ctx.value(mark))}</td><td>${ctx.fills.length}</td></tr>`
      })
      .join("")
  }

  function render() {
    const series = [{ data: s.mid, color: color("ink"), width: 2 }]
    const dots = []
    for (const name of s.names) {
      const c = swatch(name)
      series.push({ data: s.quotes[name].bid, color: c, width: 1 }, { data: s.quotes[name].ask, color: c, width: 1 })
      for (const f of s.ctx[name].fills) {
        dots.push({ x: f.t - s.start, y: f.price, up: f.side === BUY, color: c })
      }
    }
    lines($("price"), { n: s.n, series, dots, minSpan: 6, dx: DT })
    const pair = (data) => s.names.map((name) => ({ data: data[name], color: swatch(name) }))
    lines($("inventory"), { n: s.n, series: pair(s.q), minSpan: 4, dx: DT })
    lines($("pnl"), { n: s.n, series: pair(s.pnl), minSpan: 4, dx: DT })
    $("clock").textContent = `minute ${fmt(s.minute, 1)} of ${MINUTES}`
    $("go").textContent = running ? "Pause" : s.done ? "Play again" : s.j ? "Resume" : "Start"
    for (const name of Object.keys(GAMMAS)) {
      for (const el of document.querySelectorAll(`.k.${name}`)) el.hidden = !s.names.includes(name)
    }
    ladder()
    positions()
  }

  function finish() {
    running = false
    render()
    const m = s.market
    const close = m.mark()
    $("split").innerHTML = s.names
      .map((name) => {
        const ctx = s.ctx[name]
        const { edge, inventory, pnl } = split(ctx, close)
        const front = ctx.fills.filter((f) => f.ahead === 0).length
        return `<tr><td>${NAMES[name]}</td><td>${signed(edge)}</td><td>${signed(inventory)}</td><td>${signed(pnl)}</td><td>${ctx.fills.length}</td><td>${front}</td></tr>`
      })
      .join("")
    $("queue").innerHTML = [0, 1, 2, 3]
      .map((k) => {
        const cells = s.names.map((name) => {
          const r = byQueue(s.ctx[name], m.tape)[k]
          return `<td>${r.n}</td><td>${r.n ? signed(r.markout, 2) : ""}</td>`
        })
        return `<tr><td>${k === 3 ? "3 or more" : k}</td>${cells.join("")}</tr>`
      })
      .join("")
    $("queue-head").innerHTML =
      "<th>ahead at entry</th>" +
      s.names.map((n) => `<th>${NAMES[n]} fills</th><th>1-min markout</th>`).join("")
    const key = JSON.stringify(state.cfg)
    if (!others.has(key)) others.set(key, distribution(state.cfg, 400))
    const dist = others.get(key)
    const lead = s.names.map((name) => {
      const pnl = split(s.ctx[name], close).pnl
      const xs = dist[name]
      return `${NAMES[name]} made ${fmt(pnl)} ticks, better than ${fmt(100 * rank(xs, pnl), 0)}% of 400 other sessions with the same settings (mean ${fmt(mean(xs))}, sd ${fmt(sd(xs))}).`
    })
    $("lead").textContent = lead.join(" ")
    $("results").hidden = false
    const all = s.names.flatMap((n) => dist[n])
    const lo = Math.floor(Math.min(...all) / 5) * 5
    const hi = Math.ceil(Math.max(...all) / 5) * 5
    for (const name of Object.keys(GAMMAS)) {
      const box = $(`dist-${name}`)
      box.hidden = !s.names.includes(name)
      if (box.hidden) continue
      const pnl = split(s.ctx[name], close).pnl
      bars(box.querySelector("canvas"), {
        counts: histogram(dist[name], lo, hi, 30),
        lo,
        hi,
        marks: [{ x: pnl, label: "this session", color: swatch(name) }],
      })
    }
  }

  $("go").addEventListener("click", toggle)
  $("skip").addEventListener("click", skip)
  $("seed").addEventListener("change", (e) => {
    state.seed = Math.max(1, Math.floor(Number(e.target.value)) || 1)
    reset()
  })
  $("reseed").addEventListener("click", () => {
    state.seed = 1 + Math.floor(Math.random() * 99999)
    reset()
  })
  for (const name of Object.keys(GAMMAS)) {
    $(`${name}-gamma`).addEventListener("change", (e) => {
      state.cfg[name] = e.target.value === "off" ? null : Number(e.target.value)
      reset()
    })
  }
  document.addEventListener("keydown", (e) => {
    if (e.key !== " " || e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return
    e.preventDefault()
    toggle()
  })
  window.addEventListener("resize", () => (s.done && !$("results").hidden ? finish() : render()))
  reset()
}

if (typeof document !== "undefined") page()
