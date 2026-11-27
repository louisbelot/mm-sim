import { quotes as asQuotes } from "./as.js"
import { BUY, SELL } from "./book.js"
import { Model } from "./glft.js"

export function others(book, side, px, owner) {
  const l = book.level(side, px)
  if (l === null) return 0
  let n = 0
  for (const o of l.orders) if (o.owner !== owner) n += o.qty
  return n
}

class Quoter {
  constructor({ size = 1, maxPosition = null, keep = 0, inside = true, thin = 0, back = 1 } = {}) {
    this.size = size
    this.maxPosition = maxPosition
    this.keep = keep
    this.inside = inside
    this.thin = thin
    this.back = back
    this.ids = [null, null]
  }

  onStart(ctx) {
    this.requote(ctx)
  }

  onBook(ctx) {
    this.requote(ctx)
  }

  onFill(ctx) {
    this.requote(ctx)
  }

  requote(ctx) {
    let [bid, ask] = this.targets(ctx)
    if (this.thin) {
      bid = this.stepBack(ctx, BUY, bid)
      ask = this.stepBack(ctx, SELL, ask)
    }
    if (!this.inside) {
      if (bid !== null && ctx.bid !== null) bid = Math.min(bid, ctx.bid)
      if (ask !== null && ctx.ask !== null) ask = Math.max(ask, ctx.ask)
    }
    if (this.maxPosition !== null) {
      if (ctx.position + this.size > this.maxPosition) bid = null
      if (ctx.position - this.size < -this.maxPosition) ask = null
    }
    this.quote(ctx, BUY, bid)
    this.quote(ctx, SELL, ask)
  }

  stepBack(ctx, side, px) {
    if (px === null) return null
    const away = side === BUY ? -1 : 1
    for (let i = 0; i < this.back; i++) {
      if (others(ctx.book, side, px, ctx.owner) >= this.thin) break
      px += away
    }
    return px
  }

  quote(ctx, side, px) {
    const id = this.ids[side]
    const live = id === null ? undefined : ctx.orders.get(id)
    if (live !== undefined && px !== null) {
      const behind = side === BUY ? px - live.price : live.price - px
      if (behind >= 0 && behind <= this.keep) return
    }
    if (live !== undefined) ctx.cancel(id)
    this.ids[side] = px === null ? null : ctx.submit(side, px, this.size, { postOnly: true })
  }
}

export class Touch extends Quoter {
  constructor(opts = {}) {
    super(opts)
    this.half = opts.half ?? 0.5
  }

  targets(ctx) {
    const mid = ctx.mid
    if (mid === null) return [null, null]
    return [
      Math.min(Math.floor(mid - this.half), ctx.ask - 1),
      Math.max(Math.ceil(mid + this.half), ctx.bid + 1),
    ]
  }
}

export class AvellanedaStoikov extends Quoter {
  constructor({ gamma, sigma, k, ...opts }) {
    super(opts)
    Object.assign(this, { gamma, sigma, k })
  }

  targets(ctx) {
    const s = ctx.mid
    if (s === null) return [null, null]
    const [bid, ask] = asQuotes(s, ctx.position, ctx.end - ctx.t, this.gamma, this.sigma, this.k)
    return [Math.min(Math.floor(bid), ctx.ask - 1), Math.max(Math.ceil(ask), ctx.bid + 1)]
  }
}

export class GLFT extends Quoter {
  constructor(params, opts = {}) {
    const model = params instanceof Model ? params : new Model(params)
    super({ ...opts, maxPosition: model.Q * (opts.size ?? 1) })
    this.model = model
    const { bids, asks } = model.quotes()
    this.bids = bids
    this.asks = asks
  }

  targets(ctx) {
    const s = ctx.mid
    if (s === null) return [null, null]
    const Q = this.model.Q
    const i = Q + Math.min(Math.max(Math.round(ctx.position / this.size), -Q), Q)
    const db = this.bids[i]
    const da = this.asks[i]
    return [
      Number.isNaN(db) ? null : Math.min(Math.floor(s - db), ctx.ask - 1),
      Number.isNaN(da) ? null : Math.max(Math.ceil(s + da), ctx.bid + 1),
    ]
  }
}

