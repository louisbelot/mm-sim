function sturmBelow(d, e2, x) {
  let count = 0
  let p = d[0] - x
  if (p < 0) count++
  for (let i = 1; i < d.length; i++) {
    p = d[i] - x - e2 / (p === 0 ? 1e-300 : p)
    if (p < 0) count++
  }
  return count
}

function thomas(d, e, b) {
  const n = d.length
  const c = new Float64Array(n)
  const x = new Float64Array(n)
  let w = d[0]
  x[0] = b[0] / w
  for (let i = 1; i < n; i++) {
    c[i - 1] = e / w
    w = d[i] - e * c[i - 1]
    x[i] = (b[i] - e * x[i - 1]) / w
  }
  for (let i = n - 2; i >= 0; i--) x[i] -= c[i] * x[i + 1]
  return x
}

export function groundState(d, e) {
  let lo = Math.min(...d) - 2 * Math.abs(e)
  let hi = Math.max(...d) + 2 * Math.abs(e)
  for (let it = 0; it < 200 && hi - lo > 1e-15 * Math.max(1, Math.abs(lo)); it++) {
    const m = (lo + hi) / 2
    if (sturmBelow(d, e * e, m) >= 1) hi = m
    else lo = m
  }
  const shift = lo - 1e-9 * Math.max(1, Math.abs(lo))
  const ds = d.map((x) => x - shift)
  let v = new Float64Array(d.length).fill(1)
  for (let it = 0; it < 6; it++) {
    v = thomas(ds, e, v)
    const top = Math.max(...v.map(Math.abs))
    v = v.map((x) => x / top)
  }
  return { lambda: lo, v }
}

export class Model {
  constructor({ sigma, A, k, gamma, Q, mu = 0, xi = 0 }) {
    Object.assign(this, { sigma, A, k, gamma, Q, mu, xi })
  }

  get alpha() {
    return (this.k * this.gamma * this.sigma ** 2) / 2
  }

  get eta() {
    return this.A * (1 + this.gamma / this.k) ** -(1 + this.k / this.gamma)
  }

  get base() {
    return Math.log1p(this.gamma / this.k) / this.gamma + this.xi / 2
  }

  bands() {
    const d = []
    for (let q = -this.Q; q <= this.Q; q++) d.push(this.alpha * q * q - this.k * this.mu * q)
    return [d, -this.eta * Math.exp((-this.k * this.xi) / 2)]
  }

  quotes() {
    const [d, e] = this.bands()
    const { v } = groundState(d, e)
    const n = 2 * this.Q + 1
    const bids = new Array(n).fill(NaN)
    const asks = new Array(n).fill(NaN)
    for (let i = 0; i < n - 1; i++) {
      const dv = Math.log(v[i + 1] / v[i]) / this.k
      bids[i] = this.base - dv
      asks[i + 1] = this.base + dv
    }
    return { bids, asks }
  }

  approx() {
    const { gamma: g, k, sigma, A } = this
    let c = Math.sqrt(((sigma ** 2 * g) / (2 * k * A)) * (1 + g / k) ** (1 + k / g))
    c *= Math.exp((k * this.xi) / 4)
    const bids = []
    const asks = []
    for (let q = -this.Q; q <= this.Q; q++) {
      const lean = q + 0.5 - this.mu / (g * sigma ** 2)
      bids.push(this.base + lean * c)
      asks.push(this.base - (lean - 1) * c)
    }
    return { bids, asks }
  }
}
