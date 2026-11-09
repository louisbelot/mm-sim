from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmsim import hy
from mmsim.etf import hedge, stepwise
from mmsim.itch import read
from mmsim.micro import Tops, tops

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLOR = {"AAPL": "#2a78d6", "MSFT": "#eb6834", "AMZN": "#1baf7a", "INTC": "#8a5cd1"}
ROOT = Path(__file__).resolve().parents[1]
ETF = "QQQ"
NAMES = sorted(p.stem for p in (ROOT / "data").glob("*.itch") if p.stem != ETF)
AM, PM = (600.0, 780.0), (780.0, 950.0)
KS = (1, 3, 5, 10, 20, len(NAMES))


def load(symbol: str) -> Tops:
    path = ROOT / "data" / f"{symbol}.tops.npz"
    if not path.exists():
        np.savez(path, **tops(read(ROOT / "data" / f"{symbol}.itch"))._asdict())
    z = np.load(path)
    return Tops(*(z[k] for k in Tops._fields))


books = {s: load(s) for s in [ETF, *NAMES]}
px = {s: (b.t, b.mid / 100) for s, b in books.items()}


def moves(dt: float, window) -> np.ndarray:
    grid = np.arange(window[0], window[1] + 1e-9, dt / 60)
    return np.diff(np.stack([hy.sample(*px[s], grid) for s in [ETF, *NAMES]], axis=1), axis=0)


def score(h, idx, r) -> float:
    left = r[:, 0] + r[:, 1:][:, idx] @ h
    return 1 - left.var() / r[:, 0].var()


print(f"{ETF} hedged with up to {len(NAMES)} of its components, fit 10:00-13:00, "
      "scored 13:00-15:50")
fits = {dt: np.cov(moves(dt, AM).T) for dt in (1, 5, 60)}
tests = {dt: moves(dt, PM) for dt in (1, 5, 60, 300)}
print("| k  | added      | fit 60 s, score 60 s | fit 60 s, score 5 min | fit 1 s, score 1 s "
      "| fit 1 s, score 60 s | fit 1 s, score 5 min |")
print("| -- | ---------- | -------------------- | --------------------- | ------------------ "
      "| ------------------- | -------------------- |")
order60, _ = stepwise(fits[60], len(NAMES))
order1, _ = stepwise(fits[1], len(NAMES))
effect = {}
for k in KS:
    h60, _ = hedge(fits[60], order60[:k])
    h1, _ = hedge(fits[1], order1[:k])
    row = [score(h60, order60[:k], tests[60]), score(h60, order60[:k], tests[300]),
           score(h1, order1[:k], tests[1]), score(h1, order1[:k], tests[60]),
           score(h1, order1[:k], tests[300])]
    effect[k] = row
    print(f"| {k:<2} | {NAMES[order60[k - 1]]:<10} | " + " | ".join(f"{v:.3f}" for v in row) + " |")
print("stepwise order at 60 s:", " ".join(NAMES[i] for i in order60[:10]))
every = list(range(len(NAMES)))
Chy = hy.matrix([px[s] for s in [ETF, *NAMES]], AM)
hh, _ = hedge(Chy, every)
print(f"all names, Hayashi-Yoshida on ticks: score 60 s {score(hh, every, tests[60]):.3f}, "
      f"score 5 min {score(hh, every, tests[300]):.3f}")

dts = np.array([0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120, 300])
whole = (AM[0], PM[1])
grid_moves = {dt: moves(dt, whole) for dt in dts}
EPPS = ["AAPL", "MSFT", "AMZN", "INTC"]
corr = {s: [np.corrcoef(grid_moves[dt][:, [0, 1 + NAMES.index(s)]].T)[0, 1] for dt in dts]
        for s in EPPS}
r2 = []
for dt in dts[dts <= 60]:
    C = np.cov(moves(dt, AM).T)
    h, _ = hedge(C, range(len(NAMES)))
    r2.append(score(h, list(range(len(NAMES))), moves(dt, PM)))

print("\ncovariance with QQQ, 10:00-13:00, as a share of the 60 s previous-tick estimate")
print("| name | HY ticks | HY 100 ms | HY 1 s | previous tick 100 ms | previous tick 1 s |")
print("| ---- | -------- | --------- | ------ | -------------------- | ----------------- |")
cut = {s: hy.between(*px[s], *AM) for s in [ETF, *EPPS]}
for s in EPPS:
    base = moves(60, AM)
    c60 = base[:, 0] @ base[:, 1 + NAMES.index(s)]
    row = [hy.covariance(*cut[ETF], *cut[s])]
    for dt in (0.1, 1.0):
        ga = np.arange(AM[0], AM[1], dt / 60)
        gb = ga[:-1] + dt / 120
        row.append(hy.covariance(ga, hy.sample(*px[ETF], ga), gb, hy.sample(*px[s], gb)))
    for dt in (0.1, 1.0):
        m = moves(dt, AM)[:, [0, 1 + NAMES.index(s)]]
        row.append(m[:, 0] @ m[:, 1])
    print(f"| {s:<4} | " + " | ".join(f"{v / c60:.3f}" for v in row) + " |")

lags = np.arange(-500, 501, 10) / 6e4
pos, neg = lags > 0, lags < 0
print("\nlead-lag on ticks, 10:00-13:00 (positive lag: QQQ moves first)")
print("| name  | peak lag (ms) | LLR   |")
print("| ----- | ------------- | ----- |")
curves, table = {}, []
for s in NAMES:
    ell, c = hy.lead_lag(*cut[ETF], *hy.between(*px[s], *AM), lags)
    llr = np.sum(c[pos] ** 2) / np.sum(c[neg] ** 2)
    curves[s] = c
    table.append((llr, s, ell))
for llr, s, ell in sorted(table, reverse=True):
    print(f"| {s:<5} | {ell * 6e4:13.0f} | {llr:5.2f} |")
leads = sum(llr < 1 for llr, _, _ in table)
print(f"{leads} of {len(NAMES)} names lead QQQ (LLR < 1)")

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

for s in EPPS:
    left.plot(dts, corr[s], color=COLOR[s], lw=1.6, marker="o", ms=3, label=f"corr(QQQ, {s})")
left.plot(dts[dts <= 60], r2, color=INK, lw=2.0, marker="s", ms=3.5,
          label=f"out-of-sample R^2, {len(NAMES)}-name hedge")
left.set_xscale("log")
left.set_ylim(0, 1)
left.set_xlabel("sampling interval (seconds)")
left.set_ylabel("correlation / share of variance hedged")
left.set_title("Epps effect: QQQ against its components", color=INK, fontsize=10)
left.legend(frameon=False, fontsize=8, loc="lower right")

ms = lags * 6e4
for s in ("AMZN", "AAPL", "INTC"):
    c = curves[s]
    right.plot(ms, c / c.max(), color=COLOR[s], lw=1.4, label=s)
right.axvline(0, color=MUTED, lw=0.8)
right.set_xlim(-300, 300)
right.set_xlabel("lag (ms), positive when QQQ moves first")
right.set_ylabel("Hayashi-Yoshida covariance / peak")
right.set_title("Who moves first: lead-lag on the tick tape", color=INK, fontsize=10)
right.legend(frameon=False, fontsize=8)

fig.tight_layout()
out = ROOT / "docs" / "etf_hedge.png"
fig.savefig(out, dpi=150, facecolor=SURFACE)
print(f"\nwrote {out}")
