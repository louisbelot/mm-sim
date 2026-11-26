import { BUY, SELL } from "./book.js"
import { expo } from "./cst.js"
import { normal } from "./rng.js"

export class Fundamental {
  constructor(sigma, rand) {
    this.sigma = sigma
    this.rand = rand
    this.t = null
    this.v = NaN
    this.path = []
  }

  start(t, v) {
    this.t = t
    this.v = v
    this.path = [[t, v]]
  }

  at(t) {
    if (t > this.t) {
      this.v += this.sigma * Math.sqrt(t - this.t) * normal(this.rand)
      this.t = t
      this.path.push([t, this.v])
    }
    return this.v
  }
}

export class InformedTrader {
  constructor(value, rate, rand, { edge = 0, size = 1 } = {}) {
    this.value = value
    this.rate = rate
    this.rand = rand
    this.edge = edge
    this.size = size
  }

  onStart(ctx) {
    this.begin(ctx)
    ctx.setTimer(expo(this.rand, this.rate))
  }

  onTimer(ctx) {
    if (this.begin(ctx)) this.trade(ctx)
    ctx.setTimer(expo(this.rand, this.rate))
  }

  onBook() {}

  onFill() {}

  trade(ctx) {
    const v = this.value.at(ctx.t)
    const { bid, ask } = ctx.book
    if (ask !== null && v > ask + this.edge) ctx.submit(BUY, 0, this.size, { market: true })
    else if (bid !== null && v < bid - this.edge) ctx.submit(SELL, 0, this.size, { market: true })
  }

  begin(ctx) {
    if (this.value.t === null) {
      const mid = ctx.book.mid()
      if (mid === null) return false
      this.value.start(ctx.t, mid)
    }
    return true
  }
}
