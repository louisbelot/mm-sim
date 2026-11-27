import { BUY, SELL } from "./book.js"

export function skyPerfect(depth = 10) {
  const lam = [1.85, 1.51, 1.09, 0.88, 0.77]
  for (let i = 6; i <= depth; i++) lam.push(1.92 / i ** 0.52)
  return { lam: lam.slice(0, depth), theta: [0.71, 0.81, 0.68, 0.56, 0.47], mu: 0.94 }
}

export function thetaAt(p, i) {
  let j = Math.min(i, p.theta.length) - 1
  if (j < 0) j += p.theta.length
  return p.theta[j]
}

export const expo = (rand, rate) => -Math.log(1 - rand()) / rate

export const ITCH_ALPHA = 2.7
export const ITCH_CAP = 20

export function zetaSizes(alpha, cap) {
  const w = []
  for (let k = 1; k <= cap; k++) w.push(k ** -alpha)
  const total = w.reduce((a, b) => a + b, 0)
  const pmf = w.map((x) => x / total)
  const cum = []
  let s = 0
  for (const p of pmf) cum.push((s += p))
  return { pmf, cum, mean: pmf.reduce((a, p, i) => a + (i + 1) * p, 0) }
}

export const itchSizes = () => zetaSizes(ITCH_ALPHA, ITCH_CAP)

export function drawSize(sizes, rand) {
  const u = rand() * sizes.cum[sizes.cum.length - 1]
  let k = 0
  while (k < sizes.cum.length - 1 && u >= sizes.cum[k]) k++
  return k + 1
}

export class ZeroIntelligence {
  constructor(params, owner = 0, sizes = null) {
    this.p = params
    this.owner = owner
    this.sizes = sizes
    this.cum = []
    let s = 0
    for (const x of params.lam) this.cum.push((s += x))
    this.bid = null
    this.ask = null
  }

  seed(book, bid, ask, ids) {
    this.p.lam.forEach((lam, j) => {
      const i = j + 1
      const n = Math.max(1, Math.round(lam / thetaAt(this.p, i)))
      for (let c = 0; c < n; c++) {
        book.submit(ids(), BUY, bid - i + 1, 1, { owner: this.owner })
        book.submit(ids(), SELL, ask + i - 1, 1, { owner: this.owner })
      }
    })
  }

  rate() {
    return this.sizes === null ? this.p.mu : this.p.mu / this.sizes.mean
  }

  menu(book) {
    const bid = book.bid
    const ask = book.ask
    if (bid !== null) this.bid = bid
    if (ask !== null) this.ask = ask
    if (this.bid === null || this.ask === null) throw new Error("the book needs a bid and an ask")

    const lam = this.cum[this.cum.length - 1]
    const menu = [
      ["limit", BUY, lam],
      ["limit", SELL, lam],
    ]
    if (ask !== null) menu.push(["market", BUY, this.rate(BUY)])
    if (bid !== null) menu.push(["market", SELL, this.rate(SELL)])
    for (const side of [BUY, SELL]) {
      for (const l of book.levels(side)) {
        let n = 0
        for (const o of l.orders) if (o.owner === this.owner) n++
        if (n) {
          const i = side === BUY ? this.ask - l.price : l.price - this.bid
          menu.push(["cancel", l, thetaAt(this.p, i) * n, n])
        }
      }
    }
    return menu
  }

  step(book, rand, ids) {
    const menu = this.menu(book)
    const total = sum(menu)
    const dt = expo(rand, total)
    return { dt, fire: this.fire(menu[pick(menu, rand() * total)], rand, ids) }
  }

  fire([kind, arg, , n], rand, ids) {
    if (kind === "cancel") {
      let j = Math.floor(rand() * n)
      for (const o of arg.orders) {
        if (o.owner === this.owner && j-- === 0) {
          const id = o.id
          return (b) => b.cancel(id)
        }
      }
    }
    const id = ids()
    if (kind === "market") {
      const qty = this.sizes === null ? 1 : drawSize(this.sizes, rand)
      return (b) => b.submit(id, arg, 0, qty, { owner: this.owner, market: true })
    }
    const lam = this.cum[this.cum.length - 1]
    const u = rand() * lam
    let i = 1
    while (i < this.cum.length && u >= this.cum[i - 1]) i++
    const px = arg === BUY ? this.ask - i : this.bid + i
    return (b) => b.submit(id, arg, px, 1, { owner: this.owner })
  }
}

function sum(menu) {
  let total = 0
  for (const m of menu) total += m[2]
  return total
}

function pick(menu, u) {
  let k = 0
  while (k < menu.length - 1 && u >= menu[k][2]) u -= menu[k++][2]
  return k
}

export function symmetricHawkes(rate, own, cross, beta) {
  const m = rate * (1 - own - cross)
  return {
    mu: [m, m],
    alpha: [
      [own * beta, cross * beta],
      [cross * beta, own * beta],
    ],
    beta: [
      [beta, beta],
      [beta, beta],
    ],
  }
}

export class HawkesFlow extends ZeroIntelligence {
  constructor(params, hawkes, owner = 0, sizes = null) {
    super(params, owner, sizes)
    this.h = hawkes
    this.t = 0
    this.s = [
      [0, 0],
      [0, 0],
    ]
    this.markets = []
  }

  rate(side, t = this.t) {
    const { mu, beta } = this.h
    const s = this.s[side]
    return mu[side] + s[0] * Math.exp(-beta[side][0] * (t - this.t)) + s[1] * Math.exp(-beta[side][1] * (t - this.t))
  }

  advance(t) {
    const { beta } = this.h
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) this.s[i][j] *= Math.exp(-beta[i][j] * (t - this.t))
    this.t = t
  }

  step(book, rand, ids, t) {
    this.advance(t)
    const menu = this.menu(book)
    const bound = sum(menu)
    let dt = 0
    for (;;) {
      dt += expo(rand, bound)
      for (const m of menu) if (m[0] === "market") m[2] = this.rate(m[1], t + dt)
      const u = rand() * bound
      if (u >= sum(menu)) continue
      const choice = menu[pick(menu, u)]
      const fire = this.fire(choice, rand, ids)
      if (choice[0] !== "market") return { dt, fire }
      const at = t + dt
      const side = choice[1]
      return {
        dt,
        fire: (b) => {
          this.advance(at)
          for (let i = 0; i < 2; i++) this.s[i][side] += this.h.alpha[i][side]
          this.markets.push([at, side])
          return fire(b)
        },
      }
    }
  }
}
