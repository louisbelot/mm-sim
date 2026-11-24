export const BUY = 0
export const SELL = 1

const SIGN = [-1, 1]

function insertSorted(keys, key) {
  let lo = 0
  let hi = keys.length
  while (lo < hi) {
    const m = (lo + hi) >> 1
    if (keys[m] < key) lo = m + 1
    else hi = m
  }
  keys.splice(lo, 0, key)
}

export class Book {
  constructor() {
    this.sides = [new Map(), new Map()]
    this.keys = [[], []]
    this.orders = new Map()
  }

  submit(id, side, price, qty, { owner = 0, market = false, postOnly = false } = {}) {
    if (!(qty > 0)) return [{ type: "rejected", id, side, reason: "qty" }]
    if (!market && !(price > 0)) return [{ type: "rejected", id, side, reason: "price" }]
    if (postOnly && market) return [{ type: "rejected", id, side, reason: "flags" }]
    if (this.orders.has(id)) return [{ type: "rejected", id, side, reason: "duplicate" }]
    if (market) price = 0
    const out = [{ type: "accepted", id, side, price, qty }]
    this.#execute({ id, owner, side, price, qty }, market, postOnly, out)
    return out
  }

  cancel(id) {
    const o = this.orders.get(id)
    if (o === undefined) return [{ type: "rejected", id, reason: "unknown" }]
    this.#remove(o)
    return [{ type: "canceled", id, side: o.side, qty: o.qty, reason: "user" }]
  }

  best(side) {
    const keys = this.keys[side]
    return keys.length ? SIGN[side] * keys[0] : null
  }

  get bid() {
    return this.best(BUY)
  }

  get ask() {
    return this.best(SELL)
  }

  mid() {
    const bid = this.bid
    const ask = this.ask
    return bid === null || ask === null ? null : (bid + ask) / 2
  }

  *levels(side) {
    const lv = this.sides[side]
    for (const k of this.keys[side]) yield lv.get(k)
  }

  level(side, price) {
    return this.sides[side].get(SIGN[side] * price) ?? null
  }

  ahead(id) {
    const o = this.orders.get(id)
    let qty = 0
    for (const other of this.level(o.side, o.price).orders) {
      if (other === o) break
      qty += other.qty
    }
    return qty
  }

  get size() {
    return this.orders.size
  }

  audit() {
    let seen = 0
    for (const side of [BUY, SELL]) {
      const keys = this.keys[side]
      const lv = this.sides[side]
      for (let i = 1; i < keys.length; i++) {
        if (keys[i - 1] >= keys[i]) return `side ${side} keys not strictly sorted`
      }
      if (keys.length !== lv.size) return `side ${side} keys and levels disagree`
      for (const k of keys) {
        const l = lv.get(k)
        if (l === undefined || l.price !== SIGN[side] * k) return `side ${side} level ${k} missing`
        if (!l.orders.length) return `side ${side} level ${l.price} is empty`
        let qty = 0
        for (const o of l.orders) {
          if (o.qty <= 0 || o.side !== side || o.price !== l.price) return `order ${o.id} bad`
          if (this.orders.get(o.id) !== o) return `order ${o.id} missing from the index`
          qty += o.qty
          seen++
        }
        if (qty !== l.qty) return `side ${side} level ${l.price} qty off`
      }
    }
    if (seen !== this.orders.size) return "index holds orders in no level"
    const bid = this.bid
    const ask = this.ask
    if (bid !== null && ask !== null && bid >= ask) return `crossed: ${bid} >= ${ask}`
    return ""
  }

  #execute(t, market, postOnly, out) {
    const opp = 1 - t.side
    const keys = this.keys[opp]
    const lv = this.sides[opp]
    const limit = SIGN[opp] * t.price
    if (postOnly && keys.length && keys[0] <= limit) {
      out.push({ type: "canceled", id: t.id, side: t.side, qty: t.qty, reason: "postOnly" })
      return
    }
    while (t.qty && keys.length && (market || keys[0] <= limit)) {
      const l = lv.get(keys[0])
      const resting = l.orders
      let n = 0
      while (t.qty && n < resting.length) {
        const m = resting[n]
        const q = Math.min(t.qty, m.qty)
        out.push({ type: "trade", id: t.id, side: t.side, price: l.price, qty: q, maker: m.id })
        t.qty -= q
        m.qty -= q
        l.qty -= q
        if (m.qty === 0) {
          this.orders.delete(m.id)
          n++
        }
      }
      resting.splice(0, n)
      if (!resting.length) {
        lv.delete(keys[0])
        keys.shift()
      }
    }
    if (t.qty === 0) return
    if (market) out.push({ type: "canceled", id: t.id, side: t.side, qty: t.qty, reason: "unfilled" })
    else this.#rest(t)
  }

  #rest(o) {
    const key = SIGN[o.side] * o.price
    const lv = this.sides[o.side]
    let l = lv.get(key)
    if (l === undefined) {
      l = { price: o.price, qty: 0, orders: [] }
      lv.set(key, l)
      insertSorted(this.keys[o.side], key)
    }
    l.orders.push(o)
    l.qty += o.qty
    this.orders.set(o.id, o)
  }

  #remove(o) {
    const key = SIGN[o.side] * o.price
    const lv = this.sides[o.side]
    const l = lv.get(key)
    l.orders.splice(l.orders.indexOf(o), 1)
    l.qty -= o.qty
    this.orders.delete(o.id)
    if (!l.orders.length) {
      lv.delete(key)
      const keys = this.keys[o.side]
      keys.splice(keys.indexOf(key), 1)
    }
  }
}
