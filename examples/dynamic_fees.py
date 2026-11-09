import math
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmsim import fees
from mmsim import venues as vn
from mmsim.amm import Pool

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6"

eth = vn.Market()
s0 = eth.sigma
reg = fees.Regimes(spike=3.0, up=1 / 20, down=1 / 4)
quoter = vn.Quoter(half=15e-4, gamma=3e-4, tau=1.0, limit=50, widen=True)
hot = replace(eth, sigma=reg.spike * s0)
hot_quoter = replace(quoter, half=reg.spike * quoter.half)
PATHS, SEED = 200, 7


def run(market, rule, q, **kw):
    return vn.run(market, 0.2, 1e6, rule, q, paths=PATHS, seed=SEED, **kw)


def se(a):
    return a.std(ddof=1) / math.sqrt(len(a))


print("ETH-like pool, +-20% around 2000 with $1M, 24 h, 60 noise trades of 1 ETH an hour")
print(f"calm vol 60%, spike vol {60 * reg.spike:.0f}%, spikes start at rate 1/20 h, last 4 h")
print("quoter: AS skew, 15 bp half spread scaled with the current vol, 50 ETH limit\n")

grids = {
    "calm": [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30],
    "spike": [8, 10, 12, 15, 18, 20, 22, 25, 30, 40, 60, 100],
    "mixed": [4, 6, 8, 10, 11, 12, 13, 14, 15, 18, 22, 30],
}
setups = {
    "calm": (eth, quoter, {}),
    "spike": (hot, hot_quoter, {}),
    "mixed": (eth, quoter, {"regimes": reg}),
}
curves, best = {}, {}
for name, grid in grids.items():
    market, q, kw = setups[name]
    ys = []
    for f in grid:
        s = run(market, f * 1e-4, q, **kw)
        ys.append((s.lp.hedged.mean(), se(s.lp.hedged)))
    curves[name] = np.array(ys)
    i = int(np.argmax(curves[name][:, 0]))
    best[name] = grid[i]
    row = " ".join(f"{f}:{y:.0f}" for f, (y, _) in zip(grid, ys, strict=True))
    print(f"{name:6s} best {grid[i]} bp ({ys[i][0]:.0f} +- {ys[i][1]:.0f})   {row}")

lo, hi = best["calm"] * 1e-4, best["spike"] * 1e-4
p = math.log(hi / lo) / math.log(reg.spike)
pool = Pool.around(eth.P0, 0.2, 1e6, 0.0)
impact = (pool.buy_target(eth.size)[0][0] ** 2 / eth.P0 - 1) * 1e4
kink = [(h * 1e4 - impact / 2) / 2 for h in (quoter.half, hot_quoter.half)]
print(f"\none noise trade moves the pool {impact:.2f} bp")
print(f"(h - impact/2)/2 = {kink[0]:.1f} and {kink[1]:.1f} bp")
print(f"per-regime optima {best['calm']} and {best['spike']} bp: gamma ~ sigma^{p:.2f}")
cut = 2 * s0
slope = (hi - lo) / ((reg.spike - 1) * s0)
lb_A = (hi - lo) / ((reg.spike**2 - 1) * s0**2)
rules = {
    f"static {best['mixed']} bp": (fees.Static(best["mixed"] * 1e-4), "market", 1.0),
    "threshold, true regime": (fees.Threshold(lo, hi, cut), "true", 1.0),
    "threshold, ewma 15 min": (fees.Threshold(lo, hi, cut), "market", 0.25),
    "threshold, ewma 1 h": (fees.Threshold(lo, hi, cut), "market", 1.0),
    "proportional, ewma 15 min": (fees.Power(0.0, lo / s0, 1.0), "market", 0.25),
    "affine, ewma 15 min": (fees.Power(lo - slope * s0, slope, 1.0), "market", 0.25),
    "quadratic, ewma 15 min": (fees.Power(lo - lb_A * s0**2, lb_A, 2.0), "market", 0.25),
    "threshold, pool price 15 min": (fees.Threshold(lo, hi, cut), "pool", 0.25),
}
print("\nmixed market, paired against the best static fee on the same 200 paths")
print(
    "rule                           mean fee  noise fees  arb fees    LVR  LP hedged (se)"
    "   gain (se)"
)
table = {}
base = None
for name, (rule, source, hl) in rules.items():
    s = run(eth, rule, quoter, regimes=reg, source=source, half_life=hl)
    h = s.lp.hedged
    base = h if base is None else base
    d = h - base
    table[name] = h.mean()
    print(
        f"{name:30s} {s.fee.mean() * 1e4:8.1f} {s.lp.noise_fees.mean():11.0f}"
        f" {s.lp.arb_fees.mean():9.0f} {s.lvr.mean():6.0f} {h.mean():10.0f} ({se(h):3.0f})"
        f" {d.mean():8.0f} ({se(d):3.0f})"
    )
print(f"\ntime in spike {s.spike.mean():.3f}, LVR in spike {(s.lvr_spike / s.lvr).mean():.3f}")

highs = (15, 18, 20, 22, 25, 30)
print("\nthreshold levels tuned directly in the mixed market (true regime), low by high")
print("     " + " ".join(f"{h:6d}" for h in highs))
for low in (4, 5, 6, 7):
    row = []
    for high in highs:
        rule = fees.Threshold(low * 1e-4, high * 1e-4, cut)
        row.append(run(eth, rule, quoter, regimes=reg, source="true").lp.hedged.mean())
    print(f"{low:4d} " + " ".join(f"{v:6.0f}" for v in row))

rng = np.random.default_rng(4)
dt = 1 / (eth.lam + eth.noise)
n = int(24 / dt)
state = np.zeros(n, dtype=bool)
for i in range(1, n):
    rate = reg.down if state[i - 1] else reg.up
    state[i] = state[i - 1] ^ (rng.random() < -math.expm1(-rate * dt))
true_sig = np.where(state, reg.spike * s0, s0)
ret = true_sig * math.sqrt(dt) * rng.standard_normal(n)
traces = {}
for hl in (0.25, 1.0):
    e = fees.Ewma(np.array([s0]), hl)
    out = np.empty(n)
    for i in range(n):
        e.update(ret[i : i + 1], dt)
        out[i] = e.sigma[0]
    traces[hl] = out
hours = np.arange(n) * dt
yearly = math.sqrt(8760) * 100

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
    }
)
fig = plt.figure(figsize=(12, 4.8), facecolor=SURFACE, layout="constrained")
gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1])
left = fig.add_subplot(gs[:, 0])
top = fig.add_subplot(gs[0, 1])
bottom = fig.add_subplot(gs[1, 1], sharex=top)
for ax in (left, top, bottom):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

colors = {"calm": AQUA, "spike": ORANGE, "mixed": BLUE}
labels = {
    "calm": "calm only, quoter 15 bp",
    "spike": "spike only, quoter 45 bp",
    "mixed": "regime switching",
}
for name, grid in grids.items():
    y, e = curves[name][:, 0], curves[name][:, 1]
    c = colors[name]
    left.plot(grid, y, color=c, linewidth=1.6, marker="o", markersize=3.5, label=labels[name])
    left.fill_between(grid, y - 2 * e, y + 2 * e, color=c, alpha=0.15, linewidth=0)
left.axhline(table["threshold, true regime"], color=VIOLET, linewidth=1.2, linestyle="--")
left.axhline(table["threshold, ewma 15 min"], color=VIOLET, linewidth=1.2, linestyle=":")
left.annotate(
    "regime switching, threshold fee\ndashed: true regime, dotted: 15 min ewma",
    (2.1, 2450),
    fontsize=9,
    color=VIOLET,
)
left.axhline(0, color=MUTED, linewidth=0.8)
left.set_xscale("log")
ticks = [2, 3, 5, 8, 12, 20, 30, 50, 100]
left.set_xticks(ticks)
left.set_xticklabels([str(t) for t in ticks])
left.set_xlim(1.8, 110)
left.set_xlabel("static pool fee (bp)")
left.set_ylabel("LP hedged P&L per day ($)")
left.set_title("Each regime has its own best fee", loc="left")
left.legend(frameon=False, loc="lower right")

top.plot(hours, true_sig * yearly, color=INK, linewidth=1.2, label="true")
top.plot(hours, traces[0.25] * yearly, color=VIOLET, linewidth=1.0, label="ewma 15 min")
top.plot(hours, traces[1.0] * yearly, color=BLUE, linewidth=1.0, label="ewma 1 h")
top.axhline(cut * yearly, color=MUTED, linewidth=0.8, linestyle="--")
top.set_ylabel("vol (%/yr)")
top.set_title("One day: the estimate and the fee it sets", loc="left")
top.legend(frameon=False, loc="upper right", ncol=3)
top.tick_params(labelbottom=False)
top.set_ylim(0, 260)

for hl, c in ((0.25, VIOLET), (1.0, BLUE)):
    bottom.plot(hours, np.where(traces[hl] > cut, hi, lo) * 1e4, color=c, linewidth=1.0)
bottom.plot(hours, np.where(state, hi, lo) * 1e4, color=INK, linewidth=1.2)
bottom.set_ylabel("fee (bp)")
bottom.set_xlabel("hour")
bottom.set_xlim(0, 24)
bottom.set_ylim(0, hi * 1e4 * 1.3)

path = Path(__file__).resolve().parent.parent / "docs" / "dynamic_fees.png"
fig.savefig(path, dpi=160, facecolor=SURFACE)
print(f"\nwrote {path}")
