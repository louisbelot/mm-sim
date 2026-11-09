import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from vollab import chains, surface, svi

from mmsim import options as o

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
ROOT = Path(__file__).resolve().parents[1]
CBOE = Path(os.environ.get("VOLLAB_CBOE", ROOT.parent / "vol-lab" / "data" / "cboe"))
DAYS = (30, 60, 91, 182)
MONEY = (0.9, 0.95, 1.0, 1.05, 1.1)
ALPHA, BETA, ETA, EPS, RHO = 0.7, 150.0, 0.1, 0.02, -0.5
LOT, GAMMA = 2.5e6, 1e-5
GAMMAS = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3)
VBAR = 4 * LOT
HALF = 0.005
DAY = 1 / 252
DT = DAY / 390
PATHS = 1000
EVERY = (1, 5, 15, 60, 390)

snap = chains.load_snapshot(chains.snapshots("SPY", CBOE)[-1])
smiles = sorted(surface.surface(snap), key=lambda s: s.T)
picked = [min(smiles, key=lambda s, d=d: abs(s.T * 365 - d)) for d in DAYS]
fits = [svi.fit_smile(s, no_butterfly=True) for s in picked]
S = snap.spot
K = np.array([round(m * S) for m in MONEY], float)
rows = [
    (k, s.T, float(np.sqrt(f.w(np.log(k / s.F)) / s.T)))
    for s, f in zip(picked, fits, strict=True)
    for k in K
]
c = o.Chain.otm(S, *map(np.array, zip(*rows, strict=True)))
z = LOT / c.g.vega
lam = o.requests(c)
print(f"SPY {snap.time:%Y-%m-%d %H:%M} UTC, spot {S:.2f}, {c.d} options")
for i in range(c.d):
    kind = "C" if c.call[i] else "P"
    print(
        f"  {c.K[i]:6.0f}{kind} {c.T[i] * 365:5.0f}d  iv {100 * c.iv[i]:5.2f}  "
        f"px {c.g.price[i]:6.2f}  vega {c.g.vega[i]:6.2f}  lot {z[i]:8.0f}  "
        f"premium {z[i] * c.g.price[i]:9.3g}  requests/day {lam[i] / 252:5.1f}"
    )


def build(gamma):
    kw = dict(lam=lam, alpha=ALPHA, beta=BETA, z=z, eta=ETA, gamma=gamma)
    rank1 = o.market(c, **kw)
    red = o.Reduced(rank1, c.g.vega, ETA, VBAR, DAY, points=161)
    return o.market(c, eps=EPS, **kw), rank1, red


def strategies(book, red):
    return {
        "portfolio vega": o.closed(book, c),
        "reduced HJB": red.policy,
        "per option": o.closed(book.independent(), c),
        "no skew": o.flat(book),
    }


book, rank1, red = build(GAMMA)
atm = int(np.argmin(np.abs(c.K - S) + 10 * np.abs(c.T * 365 - 30)))
v = np.arange(-4, 5) * LOT
rb, ra = red.quotes(DAY, v)
q = np.zeros((len(v), c.d))
q[:, atm] = v / c.g.vega[atm]
cb, ca = rank1.quotes(q, DAY)
to_vol = 100 / c.g.vega[atm]
print(f"\ngamma {GAMMA:g}, vega limit {VBAR:.3g}, reduced HJB value {red.value():,.0f}")
print("  portfolio vega   reduced bid/ask (vol pts)   closed form bid/ask")
for j in range(len(v)):
    print(
        f"  {v[j]:12.3g}     {rb[j, atm] * to_vol:7.3f} {ra[j, atm] * to_vol:7.3f}"
        f"            {cb[j, atm] * to_vol:7.3f} {ca[j, atm] * to_vol:7.3f}"
    )

sim = dict(T=DAY, dt=DT, paths=PATHS, vol=c.iv[atm], eta=ETA, eps=EPS, rho=RHO, h=HALF, vbar=VBAR)
still = {**sim, "vol": 1e-4, "eps": 0.0, "rho": 0.0, "h": 0.0, "vbar": np.inf}


def booked(tau, n, vega):
    return red.policy(tau, n, np.broadcast_to(c.g.vega, n.shape))


for name, p, kw in (
    ("model vega, no extra limit", booked, still),
    ("true vega, hard limit", red.policy, {**still, "vbar": VBAR}),
):
    r = o.simulate(c, rank1, p, seed=3, **kw)
    print(
        f"  spot frozen, {name}: simulated objective {r.objective.mean():,.0f}"
        f" ({r.objective.std() / np.sqrt(PATHS):,.0f})"
    )

runs = {name: o.simulate(c, book, p, seed=1, **sim) for name, p in strategies(book, red).items()}
print(f"\none day, {PATHS} paths, hedged every minute at {HALF} a share")
print("  strategy          mean P&L        sd    objective (se)    fills  |vega| end")
for name, r in runs.items():
    se = r.objective.std() / np.sqrt(PATHS)
    print(
        f"  {name:14s} {r.pnl.mean():11,.0f} {r.pnl.std():9,.0f} {r.objective.mean():12,.0f}"
        f" ({se:7,.0f}) {r.fills.sum(1).mean():6.1f} {np.abs(r.vega).mean():10.3g}"
    )
print("\nattribution, mean (sd)")
print("  " + " " * 14 + "".join(f"{k:>19s}" for k in o.TERMS))
for name, r in runs.items():
    cells = "".join(f"{t.mean():10,.0f} ({t.std():6,.0f})" for t in r.terms.values())
    print(f"  {name:14s}{cells}")

hedges = {e: o.simulate(c, book, o.closed(book, c), seed=2, **{**sim, "every": e}) for e in EVERY}
print("\n  hedge every (min)   delta+gamma sd   cost mean     P&L sd   objective")
for e, r in hedges.items():
    dg = r.terms["delta"] + r.terms["gamma"]
    print(
        f"  {e:10d}         {dg.std():12,.0f} {r.terms['cost'].mean():11,.0f}"
        f" {r.pnl.std():10,.0f} {r.objective.mean():11,.0f}"
    )

frontier = {}
for gm in GAMMAS:
    bk, _, rd = build(gm)
    for name, p in strategies(bk, rd).items():
        if name != "no skew" or gm == GAMMAS[0]:
            r = o.simulate(c, bk, p, seed=4, **sim)
            frontier.setdefault(name, []).append((gm, r.pnl.mean(), r.pnl.std()))
print("\nfrontier: gamma, mean P&L, sd")
for name, pts in frontier.items():
    for gm, mu, sd in pts:
        print(f"  {name:14s} {gm:7.0e} {mu:11,.0f} {sd:10,.0f}")

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
fig, ax = plt.subplots(2, 2, figsize=(11, 7.6), facecolor=SURFACE)
for a in ax.flat:
    a.set_facecolor(SURFACE)
    a.grid(color=GRID, lw=0.6)
    a.set_axisbelow(True)

a = ax[0, 0]
zb, za = book.quotes(np.zeros(c.d), DAY)
long = np.zeros(c.d)
long[atm] = 0.75 * VBAR / c.g.vega[atm]
lb, la = book.quotes(long, DAY)
shades = [BLUE, ORANGE, GREEN, MUTED]
for j, (s, f) in enumerate(zip(picked, fits, strict=True)):
    kk = np.linspace(0.86 * S, 1.14 * S, 200)
    a.plot(
        kk,
        100 * np.sqrt(f.w(np.log(kk / s.F)) / s.T),
        color=shades[j],
        lw=1.1,
        label=f"{s.T * 365:.0f}d",
    )
    sl = slice(5 * j, 5 * j + 5)
    vol, Vs = 100 * c.iv[sl], c.g.vega[sl]
    a.vlines(
        c.K[sl], vol - 100 * zb[sl] / Vs, vol + 100 * za[sl] / Vs, color=shades[j], lw=4, alpha=0.35
    )
    a.plot(c.K[sl] + 2, vol - 100 * lb[sl] / Vs, "v", color=shades[j], ms=4)
    a.plot(c.K[sl] + 2, vol + 100 * la[sl] / Vs, "^", color=shades[j], ms=4)
a.set_xlabel("strike")
a.set_ylabel("implied vol (%)")
a.set_title("SPY SVI smiles; bars: flat book, triangles: long 7.5M vega", color=INK, fontsize=9.5)
a.legend(frameon=False, fontsize=8)

a = ax[0, 1]
a.plot(v / 1e6, rb[:, atm] * to_vol, color=BLUE, lw=1.4, marker="o", ms=4, label="bid, reduced HJB")
a.plot(
    v / 1e6, ra[:, atm] * to_vol, color=ORANGE, lw=1.4, marker="o", ms=4, label="ask, reduced HJB"
)
a.plot(v / 1e6, cb[:, atm] * to_vol, color=BLUE, lw=1, ls="--", label="bid, closed form")
a.plot(v / 1e6, ca[:, atm] * to_vol, color=ORANGE, lw=1, ls="--", label="ask, closed form")
a.set_xlabel("portfolio vega (millions)")
a.set_ylabel("distance from mid (vol points)")
a.set_title(
    f"{c.K[atm]:.0f} put, {c.T[atm] * 365:.0f}d: quotes against portfolio vega",
    color=INK,
    fontsize=9.5,
)
a.legend(frameon=False, fontsize=8)

a = ax[1, 0]
for (name, pts), col in zip(frontier.items(), (BLUE, GREEN, ORANGE, MUTED), strict=True):
    _, mu, sd = map(np.array, zip(*pts, strict=True))
    style = "o" if len(pts) == 1 else "o-"
    a.plot(sd / 1e6, mu / 1e6, style, color=col, lw=1.3, ms=4, label=name)
a.set_xlabel("sd of daily P&L ($M)")
a.set_ylabel("mean daily P&L ($M)")
a.set_title(
    f"risk aversion {GAMMAS[0]:g} to {GAMMAS[-1]:g}, {PATHS} days each", color=INK, fontsize=9.5
)
a.legend(frameon=False, fontsize=8)

a = ax[1, 1]
err = [(r.terms["delta"] + r.terms["gamma"]).std() / 1e3 for r in hedges.values()]
cost = [-r.terms["cost"].mean() / 1e3 for r in hedges.values()]
a.plot(cost, err, "o-", color=BLUE, lw=1.4)
for e, x, y in zip(EVERY, cost, err, strict=True):
    a.annotate(
        f"{e} min", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8, color=MUTED
    )
a.set_xlabel(f"hedging cost per day ($k at {HALF} a share)")
a.set_ylabel("sd of delta + gamma P&L ($k)")
a.set_title("how often to delta hedge", color=INK, fontsize=9.5)

fig.tight_layout()
fig.savefig(ROOT / "docs" / "options_mm.png", dpi=150, facecolor=SURFACE)
