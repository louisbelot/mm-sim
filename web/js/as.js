import { mulberry32, normal } from "./rng.js"

export const PAPER = { s0: 100, T: 1, sigma: 2, dt: 0.005, k: 1.5, A: 140, gamma: 0.1 }

export const steps = (p) => Math.round(p.T / p.dt)

export const reservationPrice = (s, q, tau, gamma, sigma) => s - q * gamma * sigma ** 2 * tau

export const spread = (tau, gamma, sigma, k) =>
  gamma * sigma ** 2 * tau + (2 / gamma) * Math.log1p(gamma / k)

export function quotes(s, q, tau, gamma, sigma, k) {
  const r = reservationPrice(s, q, tau, gamma, sigma)
  const half = spread(tau, gamma, sigma, k) / 2
  return [r - half, r + half]
}

export const fillProb = (delta, p) => Math.min(1, p.A * Math.exp(-p.k * delta) * p.dt)

export function world(seed, p = PAPER, tipNoise = 0.5) {
  const n = steps(p)
  const rand = mulberry32(seed)
  const mid = new Float64Array(n + 1)
  const uBid = new Float64Array(n)
  const uAsk = new Float64Array(n)
  mid[0] = p.s0
  const dx = p.sigma * Math.sqrt(p.dt)
  for (let j = 0; j < n; j++) {
    uBid[j] = rand()
    uAsk[j] = rand()
    mid[j + 1] = mid[j] + (rand() < 0.5 ? dx : -dx)
  }
  const tip = mid[n] + tipNoise * normal(rand)
  return { seed, p, n, mid, uBid, uAsk, tip }
}
