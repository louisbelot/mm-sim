import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmsim import hy
from mmsim.etf import Band, Dealer, Fund, arbitrage, hedge, simulate, stepwise
from mmsim.itch import read
from mmsim.micro import Tops, tops
from mmsim.multi import Exponential

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
ROOT = Path(__file__).resolve().parents[1]
ETF = "QQQ"
NAMES = sorted(p.stem for p in (ROOT / "data").glob("*.itch") if p.stem != ETF)
AM = (600.0, 780.0)
DAY_MIN = 390
Z, FLOW, GAMMA = 5000.0, Exponential(150, 100.0), 1e-4
CLOSE = 30 / DAY_MIN
KS = (0, 1, 3, 5, 10, len(NAMES))
PATHS, DT = 2000, 1 / 3900


def load(symbol: str) -> Tops:
    path = ROOT / "data" / f"{symbol}.tops.npz"
    if not path.exists():
        np.savez(path, **tops(read(ROOT / "data" / f"{symbol}.itch"))._asdict())
    z = np.load(path)
    return Tops(*(z[k] for k in Tops._fields))


def liquidity(symbols) -> tuple[np.ndarray, np.ndarray]:
    path = ROOT / "data" / "etf_liquidity.npz"
    if path.exists():
        z = np.load(path)
        if list(z["symbols"]) == list(symbols):
            return z["volume"], z["spread"]
    volume, spread = [], []
    for s in symbols:
        msgs = read(ROOT / "data" / f"{s}.itch")
        lo, hi = 570 * 60e9, 960 * 60e9
        traded = (m.qty for m in msgs if m.kind in "ECP" and m.printable and lo <= m.t < hi)
        volume.append(sum(traded))
        b = load(s).between(570, 960)
        w = np.diff(np.r_[b.t, 960.0])
        spread.append(np.sum(b.spread * w) / np.sum(w) / 100)
    np.savez(path, symbols=symbols, volume=volume, spread=spread)
    return np.array(volume, dtype=float), np.array(spread)


symbols = [ETF, *NAMES]
grid = np.arange(AM[0], AM[1] + 1e-9, 1.0)
moves = np.diff(
    np.stack([hy.sample(load(s).t, load(s).mid / 100, grid) for s in symbols], 1), axis=0
)
C = np.cov(moves.T) * DAY_MIN
h, resid = hedge(C, range(len(NAMES)))
fund = Fund(-h, C[1:, 1:], resid)
volume, spread = liquidity(symbols)
eta = spread / volume
order, var = stepwise(C, len(NAMES))

print(f"QQQ sd {math.sqrt(C[0, 0]):.3f} $/day, residual after all {len(NAMES)} names "
      f"{math.sqrt(resid):.3f} $/day")
print(f"QQQ NASDAQ volume {volume[0]:,.0f}, mean spread {spread[0] * 100:.2f} cents, "
      f"eta {eta[0]:.2e}")
print("| k  | names added                 | sd left ($/day) | dealer's hedge removes |")
print("| -- | --------------------------- | --------------- | ---------------------- |")
for k in KS[1:]:
    names = tuple(order[:k])
    d = Dealer(fund, FLOW, GAMMA, Z, names, tuple(eta[1:][list(names)]))
    hs = d.stationary_hedge()
    c = C[1:, 0][list(names)]
    left_var = C[0, 0] + 2 * hs @ c + hs @ C[1:, 1:][np.ix_(names, names)] @ hs
    share = (C[0, 0] - left_var) / (C[0, 0] - var[k])
    tag = " ".join(NAMES[i] for i in names[:4]) + (" ..." if k > 4 else "")
    print(f"| {k:<2} | {tag:<27} | {math.sqrt(var[k]):15.3f} | {share:22.3f} |")


def run(k: int, redeem: bool, thin: float):
    names = tuple(order[:k])
    e = eta * np.r_[1.0, np.full(len(NAMES), thin)]
    d = Dealer(fund, FLOW, GAMMA, Z, names, tuple(e[1:][list(names)]),
               kappa_etf=e[0] / CLOSE, kappa=e[1:] / CLOSE, redeem=redeem)
    return simulate(d, 1.0, dt=DT, paths=PATHS, seed=7)


print(f"\none day, {PATHS} paths, {Z:.0f}-share requests, gamma {GAMMA}")
print("| k  | close   | P&L    | se  | sd     | objective | spread | hedging | close cost "
      "| fills | E abs(q_T) |")
print("| -- | ------- | ------ | --- | ------ | --------- | ------ | ------- | ---------- "
      "| ----- | ---------- |")
results = {}
for thin in (1.0, 100.0):
    for k in KS:
        for redeem in (False, True):
            r = results[k, redeem, thin] = run(k, redeem, thin)
            if thin > 1 and k not in (0, 10, len(NAMES)):
                continue
            se = r.pnl.std() / math.sqrt(PATHS)
            close = ("redeem" if redeem else "unwind") + ("*" if thin > 1 else "")
            print(f"| {k:<2} | {close:<7} | {r.pnl.mean():6.0f} | {se:3.0f} "
                  f"| {r.pnl.std():6.0f} | {r.objective.mean():9.0f} | {r.spread.mean():6.0f} "
                  f"| {r.hedging.mean():7.0f} | {r.close.mean():10.0f} | {r.fills.mean():5.1f} "
                  f"| {np.abs(r.x[:, 0]).mean():10.0f} |")
print("* components 100 times less liquid")

print("\nbreak-even redemption fee per share (redeem beats unwind while the fee is below)")
for thin in (1.0, 100.0):
    for k in (1, 10, len(NAMES)):
        u, r = results[k, False, thin], results[k, True, thin]
        gain = r.objective - u.objective
        q = np.abs(r.x[:, 0]).mean()
        print(f"liquidity x{1 / thin:g}, k {k:2}: {gain.mean() / q * 100:6.3f} cents "
              f"(se {gain.std() / math.sqrt(PATHS) / q * 100:.3f})")

kyle = 0.01 / 50_000
print(f"\nAP arbitrage of the premium: s 0.05 $/root day, checks once a minute, "
      f"impact {kyle:.0e} $/share")
print("| band c (cents) | P(trade) | sim   | trades/day | units/day | sim       "
      "| AP profit $/day | sim    | premium sd (cents) | sim   |")
print("| -------------- | -------- | ----- | ---------- | --------- | --------- "
      "| --------------- | ------ | ------------------ | ----- |")
for c in (0.0, 0.005, 0.01, 0.02, 0.05):
    b = Band(0.05, DAY_MIN, c, kyle)
    r = arbitrage(b, 1000.0, seed=11)
    print(f"| {c * 100:14.1f} | {b.p_trade:8.3f} | {r.trades / len(r.seen):5.3f} "
          f"| {b.trades:10.1f} | {b.units:9,.0f} | {r.units / r.T:9,.0f} "
          f"| {b.profit:15.1f} | {r.profit / r.T:6.1f} | {b.sd * 100:18.3f} "
          f"| {np.sqrt(np.mean(r.seen**2)) * 100:5.3f} |")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.0), facecolor=SURFACE)
for ax in (left, right):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)

ks = np.arange(len(KS))
for thin, color in ((1.0, BLUE), (100.0, ORANGE)):
    for redeem, ls, tag in ((False, "-", "unwind"), (True, "--", "redeem")):
        obj = np.array([results[k, redeem, thin].objective.mean() for k in KS])
        se = np.array([results[k, redeem, thin].objective.std() / math.sqrt(PATHS) for k in KS])
        label = f"{tag}, components {'as traded' if thin == 1 else '100x less liquid'}"
        left.errorbar(ks, obj, yerr=2 * se, color=color, ls="none", capsize=3, lw=1.0)
        left.plot(ks, obj, color=color, ls=ls, lw=1.6, marker="o", ms=3.5, label=label)
left.set_xticks(ks, [str(k) for k in KS])
left.set_xlabel("hedge names (stepwise order)")
left.set_ylabel("P&L minus risk penalty ($/day)")
left.set_title("Value of hedging QQQ in its components", color=INK, fontsize=10)
left.set_ylim(2900, 5100)
left.legend(frameon=False, fontsize=8, loc="upper left")

sd = [results[k, False, 1.0].pnl.std() for k in KS]
mean = [results[k, False, 1.0].pnl.mean() for k in KS]
right.plot(sd, mean, color=BLUE, lw=1.2, marker="o", ms=4)
for k, x, y in zip(KS, sd, mean, strict=True):
    right.annotate(str(k), (x, y), textcoords="offset points", xytext=(6, 4),
                   fontsize=8, color=MUTED)
right.set_xlabel("sd of daily P&L ($), points labelled by hedge names")
right.set_ylabel("mean daily P&L ($)")
right.set_title("Hedging buys both: more spread, less risk", color=INK, fontsize=10)

fig.tight_layout()
out = ROOT / "docs" / "etf_dealer.png"
fig.savefig(out, dpi=150, facecolor=SURFACE)
print(f"\nwrote {out}")
