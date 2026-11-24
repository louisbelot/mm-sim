const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim()

export function color(name) {
  return css(`--${name}`)
}

function setup(canvas) {
  const dpr = window.devicePixelRatio || 1
  const { width, height } = canvas.getBoundingClientRect()
  canvas.width = Math.round(width * dpr)
  canvas.height = Math.round(height * dpr)
  const g = canvas.getContext("2d")
  g.setTransform(dpr, 0, 0, dpr, 0, 0)
  g.clearRect(0, 0, width, height)
  return { g, width, height }
}

function niceTicks(lo, hi, n) {
  const raw = (hi - lo) / n
  const mag = 10 ** Math.floor(Math.log10(raw))
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw)
  const out = []
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10))
  return out
}

function frame(g, width, height, x0, x1, y0, y1, pad, xLabel) {
  const X = (x) => pad.l + ((x - x0) / (x1 - x0)) * (width - pad.l - pad.r)
  const Y = (y) => height - pad.b - ((y - y0) / (y1 - y0)) * (height - pad.t - pad.b)
  g.font = "11px ui-sans-serif, system-ui, sans-serif"
  g.lineWidth = 1
  g.strokeStyle = color("grid")
  g.fillStyle = color("muted")
  g.textAlign = "right"
  g.textBaseline = "middle"
  for (const v of niceTicks(y0, y1, 4)) {
    g.beginPath()
    g.moveTo(pad.l, Y(v))
    g.lineTo(width - pad.r, Y(v))
    g.stroke()
    g.fillText(String(v), pad.l - 6, Y(v))
  }
  g.textAlign = "center"
  g.textBaseline = "top"
  for (const v of niceTicks(x0, x1, 5)) g.fillText(xLabel(v), X(v), height - pad.b + 5)
  return { X, Y }
}

function extent(values, minSpan) {
  let lo = Infinity
  let hi = -Infinity
  for (const v of values) {
    if (!Number.isFinite(v)) continue
    lo = Math.min(lo, v)
    hi = Math.max(hi, v)
  }
  if (!Number.isFinite(lo)) return [-minSpan / 2, minSpan / 2]
  const mid = (lo + hi) / 2
  const half = Math.max(hi - lo, minSpan) / 2 * 1.1
  return [mid - half, mid + half]
}

export function lines(canvas, { n, series, dots = [], levels = [], minSpan = 1, dx = 1 }) {
  const { g, width, height } = setup(canvas)
  const all = series.flatMap((s) => Array.from(s.data))
  for (const l of levels) all.push(l.y)
  const [y0, y1] = extent(all, minSpan)
  const pad = { l: 44, r: 40, t: 8, b: 22 }
  const { X, Y } = frame(g, width, height, 0, n * dx, y0, y1, pad, String)
  for (const l of levels) {
    g.strokeStyle = l.color
    g.setLineDash([4, 4])
    g.beginPath()
    g.moveTo(pad.l, Y(l.y))
    g.lineTo(width - pad.r, Y(l.y))
    g.stroke()
    g.setLineDash([])
    g.fillStyle = l.color
    g.textAlign = "left"
    g.textBaseline = "middle"
    g.fillText(l.label, width - pad.r + 4, Y(l.y))
  }
  for (const s of series) {
    g.strokeStyle = s.color
    g.lineWidth = s.width ?? 1.5
    g.setLineDash(s.dash ?? [])
    g.beginPath()
    let pen = false
    s.data.forEach((v, i) => {
      if (!Number.isFinite(v)) pen = false
      else if (pen) g.lineTo(X(i * dx), Y(v))
      else {
        g.moveTo(X(i * dx), Y(v))
        pen = true
      }
    })
    g.stroke()
  }
  g.setLineDash([])
  for (const d of dots) {
    g.fillStyle = d.color
    g.beginPath()
    if (d.up) {
      g.moveTo(X(d.x), Y(d.y) - 5)
      g.lineTo(X(d.x) - 4, Y(d.y) + 3)
      g.lineTo(X(d.x) + 4, Y(d.y) + 3)
    } else {
      g.moveTo(X(d.x), Y(d.y) + 5)
      g.lineTo(X(d.x) - 4, Y(d.y) - 3)
      g.lineTo(X(d.x) + 4, Y(d.y) - 3)
    }
    g.fill()
  }
}

export function bars(canvas, { counts, lo, hi, marks }) {
  const { g, width, height } = setup(canvas)
  const pad = { l: 44, r: 12, t: 22, b: 22 }
  const top = Math.max(...counts) * 1.15
  const { X, Y } = frame(g, width, height, lo, hi, 0, top, pad, (v) => String(v))
  const w = (hi - lo) / counts.length
  g.fillStyle = color("bar")
  counts.forEach((c, i) => {
    const x = X(lo + i * w)
    g.fillRect(x + 0.5, Y(c), X(lo + (i + 1) * w) - x - 1, Y(0) - Y(c))
  })
  g.textBaseline = "bottom"
  for (const m of marks) {
    const x = X(Math.max(lo, Math.min(hi, m.x)))
    g.strokeStyle = m.color
    g.lineWidth = 2
    g.beginPath()
    g.moveTo(x, Y(0))
    g.lineTo(x, pad.t)
    g.stroke()
    g.fillStyle = m.color
    g.textAlign = m.align ?? "center"
    g.fillText(m.label, x, pad.t - 4)
  }
}
