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

export class ZeroIntelligence {
  constructor(params, owner = 0) {
    this.p = params
    this.owner = owner
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

  step(book, rand, ids) {
    const p = this.p
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
    if (ask !== null) menu.push(["market", BUY, p.mu])
    if (bid !== null) menu.push(["market", SELL, p.mu])
    for (const side of [BUY, SELL]) {
      for (const l of book.levels(side)) {
        let n = 0
        for (const o of l.orders) if (o.owner === this.owner) n++
        if (n) {
          const i = side === BUY ? this.ask - l.price : l.price - this.bid
          menu.push(["cancel", l, thetaAt(p, i) * n, n])
        }
      }
    }
    let total = 0
    for (const m of menu) total += m[2]
    const dt = expo(rand, total)
    let u = rand() * total
    let k = 0
    while (k < menu.length - 1 && u >= menu[k][2]) u -= menu[k++][2]
    return { dt, fire: this.#fire(menu[k], rand, ids) }
  }

  #fire([kind, arg, , n], rand, ids) {
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
    if (kind === "market") return (b) => b.submit(id, arg, 0, 1, { owner: this.owner, market: true })
    const lam = this.cum[this.cum.length - 1]
    const u = rand() * lam
    let i = 1
    while (i < this.cum.length && u >= this.cum[i - 1]) i++
    const px = arg === BUY ? this.ask - i : this.bid + i
    return (b) => b.submit(id, arg, px, 1, { owner: this.owner })
  }
}
