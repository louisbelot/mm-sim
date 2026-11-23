import { PAPER, world } from "./as.js"
import { bars, color, lines } from "./chart.js"
import { Session, distribution, marks, split } from "./session.js"
import { histogram, mean, rank, sd } from "./stats.js"

const $ = (id) => document.getElementById(id)
const fmt = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : "")
const signed = (v, d = 2) => (v >= 0 ? "+" : "") + v.toFixed(d)
const SPEEDS = { slow: 4, normal: 8, fast: 20 }

let botSessions = null
let s = null
let running = false
let last = 0
let acc = 0

const params = new URLSearchParams(location.search)
const state = {
  mode: params.get("mode") === "take" ? "take" : "make",
  seed: Number(params.get("seed")) || 1 + Math.floor(Math.random() * 99999),
  insider: params.get("insider") === "1",
}

const showTip = () => state.insider && state.mode === "take"

function reset() {
  running = false
  acc = 0
  s = new Session(world(state.seed), state.mode)
  $("seed").value = state.seed
  $("half").value = s.half
  $("lean").value = 0
  document.body.dataset.mode = state.mode
  for (const b of document.querySelectorAll("[data-pick]")) {
    b.setAttribute("aria-pressed", String(b.dataset.pick === state.mode))
  }
  $("insider").checked = state.insider
  $("results").hidden = true
  const q = new URLSearchParams({ mode: state.mode, seed: state.seed })
  if (state.insider) q.set("insider", "1")
  history.replaceState(null, "", `?${q}`)
  setQuotes()
}

function tick(now) {
  if (!running) return
  acc += ((now - last) / 1000) * SPEEDS[$("speed").value]
  last = now
  while (acc >= 1 && !s.done) {
    s.advance()
    acc -= 1
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

function setQuotes() {
  s.half = Math.max(0.05, Number($("half").value))
  s.lean = Number($("lean").value)
  $("halfv").textContent = fmt(s.half)
  $("leanv").textContent = signed(s.lean, 1)
  render()
}

function nudge(id, by) {
  const el = $(id)
  el.value = Math.min(Number(el.max), Math.max(Number(el.min), Number(el.value) + by))
  setQuotes()
}

function take(side) {
  if (!running || s.take(side) === null) return
  render()
}

function ladder() {
  if (s.done) {
    $("ladder").innerHTML = ""
    return
  }
  const [bid, ask] = s.botQuotes()
  const rows = [
    ["bot ask", ask, "ask"],
    ["mid", s.s, "mid"],
    ["bot bid", bid, "bid"],
  ]
  if (state.mode === "make") {
    const [b, a] = s.yourQuotes()
    rows.push(["your ask", a, "you"], ["your bid", b, "you"])
  }
  if (showTip()) rows.push(["tip", s.w.tip, "tip"])
  rows.sort((x, y) => y[1] - x[1])
  $("ladder").innerHTML = rows
    .map(
      ([name, px, cls]) =>
        `<tr class="${cls}"><td>${name}</td><td>${fmt(px)}</td><td>${signed(px - s.s)}</td></tr>`,
    )
    .join("")
}

function positions() {
  const row = (name, m) =>
    `<tr><td>${name}</td><td>${m.q}</td><td>${fmt(m.x)}</td><td>${fmt(m.wealth(s.s))}</td><td>${m.trades.length}</td></tr>`
  $("book").innerHTML = row("you", s.you) + row("bot", s.bot)
}

function render() {
  const n = s.w.n
  const j = s.j
  const [bid, ask] = s.done ? [NaN, NaN] : s.botQuotes()
  const series = [
    { data: [...s.bot.asks, ask], color: color("ask") },
    { data: [...s.bot.bids, bid], color: color("bid") },
    { data: s.w.mid.slice(0, j + 1), color: color("ink"), width: 2 },
  ]
  if (state.mode === "make") {
    const [b, a] = s.done ? [NaN, NaN] : s.yourQuotes()
    series.push(
      { data: [...s.you.asks, a], color: color("you"), dash: [5, 3] },
      { data: [...s.you.bids, b], color: color("you"), dash: [5, 3] },
    )
  }
  const dots = s.you.trades.map((t) => ({ x: t.j, y: t.px, up: t.side > 0, color: color("you") }))
  const levels = showTip() ? [{ y: s.w.tip, label: "tip", color: color("tip") }] : []
  lines($("price"), { n, series, dots, levels, minSpan: 3 })
  const pair = (bot, you) => [
    { data: bot, color: color("bot") },
    { data: you, color: color("you") },
  ]
  lines($("inventory"), { n, series: pair(s.bot.qPath, s.you.qPath), minSpan: 4 })
  lines($("pnl"), {
    n,
    series: pair(marks(s.bot, s.w.mid, j), marks(s.you, s.w.mid, j)),
    minSpan: 4,
  })
  $("clock").textContent = `step ${j} of ${n}, t = ${fmt(j * s.w.p.dt, 3)}`
  $("go").textContent = running ? "Pause" : s.done ? "Play again" : j ? "Resume" : "Start"
  ladder()
  positions()
}

function splitRow(name, m) {
  const { edge, inventory } = split(m, s.w.mid)
  return `<tr><td>${name}</td><td>${signed(edge)}</td><td>${signed(inventory)}</td><td>${signed(edge + inventory)}</td></tr>`
}

function summary(end) {
  const you = s.you.wealth(end)
  const bot = s.bot.wealth(end)
  const m = mean(botSessions)
  const v = sd(botSessions)
  if (state.mode === "make") {
    const verdict = you > bot ? "You won this one." : "The bot won this one."
    const pct = fmt(100 * rank(botSessions, you), 0)
    return {
      lead: `You made ${fmt(you)} and the bot made ${fmt(bot)} on the same prices and the same order arrivals. ${verdict} Over 2000 other sessions the bot averages ${fmt(m)} with a standard deviation of ${fmt(v)}; your result beats ${pct}% of them.`,
      rows: splitRow("you", s.you) + splitRow("bot", s.bot),
      marks: [
        { x: bot, label: "bot", color: color("bot") },
        { x: you, label: "you", color: color("you") },
      ],
    }
  }
  const alone = s.alone()
  const solo = alone.wealth(end)
  const effect = bot >= solo ? `gave it ${fmt(bot - solo)}` : `took ${fmt(solo - bot)} from it`
  return {
    lead: `You made ${fmt(you)} from ${s.you.trades.length} trades. The bot made ${fmt(bot)} with you in the market and would have made ${fmt(solo)} without you, so you ${effect}.`,
    rows: splitRow("you", s.you) + splitRow("bot", s.bot) + splitRow("bot without you", alone),
    marks: [
      { x: bot, label: "bot", color: color("bot") },
      { x: solo, label: "bot alone", color: color("muted") },
    ],
  }
}

function finish() {
  running = false
  botSessions ??= Array.from(distribution(2000, PAPER, 1_000_000))
  const out = summary(s.w.mid[s.w.n])
  $("lead").textContent = out.lead
  $("split").innerHTML = out.rows
  $("results").hidden = false
  const m = mean(botSessions)
  const v = sd(botSessions)
  const xs = out.marks.map((k) => k.x)
  const lo = Math.floor(Math.min(m - 4 * v, ...xs) / 5) * 5
  const hi = Math.ceil(Math.max(m + 4 * v, ...xs) / 5) * 5
  out.marks.sort((a, b) => a.x - b.x)
  out.marks[0].align = "right"
  out.marks[1].align = "left"
  bars($("dist"), { counts: histogram(botSessions, lo, hi, 40), lo, hi, marks: out.marks })
  render()
}

for (const b of document.querySelectorAll("[data-pick]")) {
  b.addEventListener("click", () => {
    state.mode = b.dataset.pick
    reset()
  })
}
$("go").addEventListener("click", toggle)
$("half").addEventListener("input", setQuotes)
$("lean").addEventListener("input", setQuotes)
$("buy").addEventListener("click", () => take(1))
$("sell").addEventListener("click", () => take(-1))
$("insider").addEventListener("change", (e) => {
  state.insider = e.target.checked
  reset()
})
$("seed").addEventListener("change", (e) => {
  state.seed = Math.max(1, Math.floor(Number(e.target.value)) || 1)
  reset()
})
$("reseed").addEventListener("click", () => {
  state.seed = 1 + Math.floor(Math.random() * 99999)
  reset()
})

const keys = {
  " ": toggle,
  ArrowUp: () => nudge("half", 0.05),
  ArrowDown: () => nudge("half", -0.05),
  ArrowLeft: () => nudge("lean", 0.1),
  ArrowRight: () => nudge("lean", -0.1),
  b: () => take(1),
  s: () => take(-1),
}
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" && e.target.type !== "range") return
  const f = keys[e.key]
  if (!f) return
  e.preventDefault()
  f()
})
window.addEventListener("resize", () => (s.done && !$("results").hidden ? finish() : render()))

reset()
