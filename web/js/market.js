import { BUY, Book, SELL } from "./book.js"

function bestOfOthers(book, side, owner) {
  for (const l of book.levels(side)) {
    for (const o of l.orders) if (o.owner !== owner) return l.price
  }
  return null
}

class Context {
  constructor(market, agent, owner) {
    this.market = market
    this.agent = agent
    this.owner = owner
    this.orders = new Map()
    this.position = 0
    this.cash = 0
    this.fills = []
  }

  get t() {
    return this.market.t
  }

  get end() {
    return this.market.end
  }

  get book() {
    return this.market.book
  }

  get bid() {
    return bestOfOthers(this.market.book, BUY, this.owner)
  }

  get ask() {
    return bestOfOthers(this.market.book, SELL, this.owner)
  }

  get mid() {
    const bid = this.bid
    const ask = this.ask
    return bid === null || ask === null ? null : (bid + ask) / 2
  }

  value(mark) {
    return this.cash + this.position * mark
  }

  submit(side, price, qty, { market = false, postOnly = false } = {}) {
    const m = this.market
    const id = m.ids()
    this.orders.set(id, { side, price: market ? 0 : price, qty, ahead: 0, canceling: false })
    const owner = this.owner
    m.execute(this, (b) => b.submit(id, side, price, qty, { owner, market, postOnly }))
    return id
  }

  cancel(id) {
    const o = this.orders.get(id)
    if (o === undefined || o.canceling) return
    o.canceling = true
    this.market.execute(this, (b) => b.cancel(id))
  }

  reduce(id, qty) {
    const o = this.orders.get(id)
    o.qty -= qty
    if (o.qty === 0) this.orders.delete(id)
  }
}

export class Market {
  constructor(flow, agents, rand) {
    this.book = new Book()
    this.flow = flow
    this.rand = rand
    this.t = 0
    this.end = 0
    this.started = false
    this.contexts = agents.map((a, i) => new Context(this, a, flow.owner + 1 + i))
    let n = 1
    this.ids = () => n++
    this.resting = new Map()
    this.pending = []
    this.busy = false
    this.preMid = null
    this.tape = []
    this.next = null
  }

  seed(bid, ask) {
    this.flow.seed(this.book, bid, ask, this.ids)
    this.tape.push([this.t, this.book.mid()])
  }

  warmUp(duration) {
    this.loop(this.t + duration)
  }

  start(duration) {
    this.end = this.t + duration
    this.started = true
    for (const ctx of this.contexts) this.queue(() => ctx.agent.onStart(ctx))
    this.drain()
  }

  loop(until) {
    for (;;) {
      if (this.next === null) {
        const { dt, fire } = this.flow.step(this.book, this.rand, this.ids)
        this.next = { at: this.t + dt, fire }
      }
      if (this.next.at > until) {
        this.t = until
        return
      }
      const { at, fire } = this.next
      this.next = null
      this.t = at
      this.execute(null, fire)
    }
  }

  mark() {
    return this.tape[this.tape.length - 1][1]
  }

  execute(sender, cmd) {
    if (sender !== null) this.next = null
    this.preMid = this.book.mid()
    const events = cmd(this.book)
    for (const e of events) {
      if (e.type === "trade") {
        const maker = this.resting.get(e.maker)
        const taker = this.resting.get(e.id)
        if (maker !== undefined) this.fill(maker, e.maker, 1 - e.side, e, true)
        if (taker !== undefined) this.fill(taker, e.id, e.side, e, false)
      } else if (sender !== null) this.bookEvent(sender, e)
    }
    const mid = this.book.mid()
    if (mid !== null) this.tape.push([this.t, mid])
    if (this.started && sender === null) {
      for (const ctx of this.contexts) this.queue(() => ctx.agent.onBook(ctx))
    }
    this.drain()
  }

  bookEvent(ctx, e) {
    if (e.type === "accepted") {
      const ahead = this.book.orders.has(e.id) ? this.book.ahead(e.id) : 0
      this.resting.set(e.id, { ctx, qty: e.qty, ahead })
      ctx.orders.get(e.id).ahead = ahead
    } else if (e.type === "canceled") {
      this.retire(e.id, e.qty)
      ctx.reduce(e.id, e.qty)
    } else if (e.type === "rejected" && ctx.orders.has(e.id)) {
      ctx.orders.delete(e.id)
    }
  }

  fill(r, id, side, e, maker) {
    const ctx = r.ctx
    const sign = side === BUY ? 1 : -1
    ctx.position += sign * e.qty
    ctx.cash -= sign * e.price * e.qty
    const lvl = maker ? this.book.level(side, e.price) : null
    const f = {
      t: this.t,
      id,
      side,
      price: e.price,
      qty: e.qty,
      maker,
      mid: this.preMid,
      ahead: maker ? r.ahead : null,
      left: maker ? (lvl === null ? 0 : lvl.qty) : null,
    }
    ctx.fills.push(f)
    this.retire(id, e.qty)
    ctx.reduce(id, e.qty)
    this.queue(() => ctx.agent.onFill(ctx, f))
  }

  retire(id, qty) {
    const r = this.resting.get(id)
    r.qty -= qty
    if (r.qty === 0) this.resting.delete(id)
  }

  queue(fn) {
    this.pending.push(fn)
  }

  drain() {
    if (this.busy) return
    this.busy = true
    for (let i = 0; i < this.pending.length; i++) this.pending[i]()
    this.pending.length = 0
    this.busy = false
  }
}


export function open(flow, makers, rand, { minutes = 60, warm = 30 } = {}) {
  const names = Object.keys(makers)
  for (let i = names.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1))
    ;[names[i], names[j]] = [names[j], names[i]]
  }
  const market = new Market(flow, names.map((n) => makers[n]()), rand)
  market.seed(10_000, 10_001)
  market.warmUp(warm)
  market.start(minutes)
  const ctx = Object.fromEntries(names.map((n, i) => [n, market.contexts[i]]))
  return { market, ctx }
}

export function session(flow, makers, rand, opts = {}) {
  const { market, ctx } = open(flow, makers, rand, opts)
  market.loop(market.end)
  const close = market.mark()
  const results = {}
  for (const [n, c] of Object.entries(ctx)) {
    results[n] = { ctx: c, pnl: c.value(close), fills: c.fills.length, position: c.position }
  }
  return { market, results }
}
