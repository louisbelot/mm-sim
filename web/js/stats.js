export const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length

export function sd(xs) {
  const m = mean(xs)
  return Math.sqrt(xs.reduce((a, x) => a + (x - m) ** 2, 0) / xs.length)
}

export const rank = (xs, x) => xs.filter((y) => y < x).length / xs.length

export function histogram(xs, lo, hi, bins) {
  const counts = new Array(bins).fill(0)
  const w = (hi - lo) / bins
  for (const x of xs) {
    const i = Math.floor((x - lo) / w)
    counts[Math.max(0, Math.min(bins - 1, i))]++
  }
  return counts
}
