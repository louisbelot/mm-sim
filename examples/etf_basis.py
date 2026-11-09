from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmsim import hy
from mmsim.etf import hedge
from mmsim.itch import read
from mmsim.micro import Tops, tops

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
ROOT = Path(__file__).resolve().parents[1]
ETF = "QQQ"
NAMES = sorted(p.stem for p in (ROOT / "data").glob("*.itch") if p.stem != ETF)
AM, PM = (600.0, 780.0), (780.0, 950.0)
THETAS = (0.5, 1.0, 2.0)
KS = np.array([1, 2, 5, 10, 20, 30, 60, 120, 300])
TAU = 10


def load(symbol: str) -> Tops:
    path = ROOT / "data" / f"{symbol}.tops.npz"
    if not path.exists():
        np.savez(path, **tops(read(ROOT / "data" / f"{symbol}.itch"))._asdict())
    z = np.load(path)
    return Tops(*(z[k] for k in Tops._fields))


def weighted(b: Tops) -> np.ndarray:
    q = b.bq + b.aq
    return np.where(q > 0, (b.bid * b.aq + b.ask * b.bq) / np.where(q > 0, q, 1), b.mid) / 100


def moves(px, dt: float, window) -> np.ndarray:
    grid = np.arange(window[0], window[1] + 1e-9, dt / 60)
    return np.diff(np.stack([hy.sample(*px[s], grid) for s in [ETF, *NAMES]], axis=1), axis=0)


def score(h, r) -> float:
    return 1 - (r[:, 0] + r[:, 1:] @ h).var() / r[:, 0].var()


def lags(x, lo: int, hi: int, pad: int = 5) -> np.ndarray:
    n = len(x)
    return np.stack([x[pad - k : n - pad - k] for k in range(lo, hi + 1)], axis=1)


def r2(ya, Xa, yp, Xp) -> float:
    b, *_ = np.linalg.lstsq(Xa, ya, rcond=None)
    return 1 - (yp - Xp @ b).var() / yp.var()


def basis(e, tau: float) -> np.ndarray:
    B = np.cumsum(e)
    m, acc = np.empty_like(B), B[0]
    for i, v in enumerate(B):
        acc += (v - acc) / tau
        m[i] = acc
    return B - m


books = {s: load(s) for s in [ETF, *NAMES]}
px = {s: (b.t, b.mid / 100) for s, b in books.items()}
every = list(range(len(NAMES)))
base = np.cov(moves(px, 60, AM).T) * (AM[1] - AM[0])
ticks = np.array([len(hy.between(*px[s], *AM)[0]) for s in NAMES])
Chy = hy.matrix([px[s] for s in [ETF, *NAMES]], AM)
P = {th: hy.matrix([px[s] for s in [ETF, *NAMES]], AM, theta=th) for th in THETAS}

print(f"{ETF} and {len(NAMES)} components, 10:00-13:00, as a share of the 60 s previous-tick value")
print("| estimator | cross cov, median | IQR | variance, median |")
print("| --------- | ----------------- | --- | ---------------- |")
rows = [("HY on ticks", Chy)] + [(f"pre-averaged, theta {th:g}", P[th]) for th in THETAS]
for name, C in rows:
    c = C[0, 1:] / base[0, 1:]
    v = np.diag(C) / np.diag(base)
    q1, q3 = np.percentile(c, [25, 75])
    print(f"| {name} | {np.median(c):.3f} | {q1:.3f}-{q3:.3f} | {np.median(v):.3f} |")
for s in ("MSFT", "AAPL", "AMZN", "INTC", "CSCO"):
    i = 1 + NAMES.index(s)
    print(
        f"  {s:5s} {ticks[i - 1]:6d} ticks: HY {Chy[0, i] / base[0, i]:.3f}, "
        f"pre-averaged {P[1.0][0, i] / base[0, i]:.3f}"
    )

tests = {dt: moves(px, dt, PM) for dt in (1, 60, 300)}
fits = {
    "previous tick 60 s": base,
    "previous tick 1 s": np.cov(moves(px, 1, AM).T),
    "HY on ticks": Chy,
    "pre-averaged, raw": P[1.0],
    "pre-averaged, eigenvalues >= 1e-3 max": hy.psd(P[1.0], 1e-3),
}
print(f"\nall {len(NAMES)} names, fit 10:00-13:00, share of {ETF} variance hedged 13:00-15:50")
print("| covariance | score 1 s | score 60 s | score 5 min |")
print("| ---------- | --------- | ---------- | ----------- |")
for name, C in fits.items():
    h, _ = hedge(C, every)
    print(f"| {name} | " + " | ".join(f"{score(h, tests[dt]):.3f}" for dt in (1, 60, 300)) + " |")
w = np.linalg.eigvalsh(P[1.0])
print(
    f"pre-averaged matrix: {np.sum(w < 0)} negative eigenvalues, "
    f"smallest {w.min() / w.max():.3f} of the largest"
)

h60, _ = hedge(base, every)
curves = {}
print(f"\none-second {ETF} moves against the 60 s hedge basket, fit 10:00-13:00, R^2 13:00-15:50")
print("| prices | same second | + 5 s before | + 5 s after | both | acf QQQ | acf basket |")
print("| ------ | ----------- | ------------ | ----------- | ---- | ------- | ---------- |")
for label, prices in (("mid", px), ("size-weighted mid", None)):
    if prices is None:
        prices = {s: (b.t, weighted(b)) for s, b in books.items()}
    ra, rp = moves(prices, 1, AM), moves(prices, 1, PM)
    ya, xa, yp, xp = ra[:, 0], -(ra[:, 1:] @ h60), rp[:, 0], -(rp[:, 1:] @ h60)
    row = [
        r2(ya[5:-5], lags(xa, lo, hi), yp[5:-5], lags(xp, lo, hi))
        for lo, hi in ((0, 0), (0, 5), (-5, 0), (-5, 5))
    ]
    acf = [np.corrcoef(v[1:], v[:-1])[0, 1] for v in (yp, xp)]
    print(f"| {label} | " + " | ".join(f"{v:.3f}" for v in row + acf) + " |")
    beta = ya @ xa / (xa @ xa)
    e = yp - beta * xp
    B = np.cumsum(e)
    curves[label] = [np.var(B[k:] - B[:-k]) / (k * e.var()) for k in KS]
    za, zp = basis(ya - beta * xa, TAU), basis(e, TAU)
    Xa = np.stack([ya[:-1], za[:-1]], axis=1)
    Xp = np.stack([yp[:-1], zp[:-1]], axis=1)
    own = r2(ya[1:], Xa[:, :1], yp[1:], Xp[:, :1])
    both = r2(ya[1:], Xa, yp[1:], Xp)
    print(
        f"  {label}: basis variance ratio "
        + ", ".join(f"{k} s {v:.2f}" for k, v in zip(KS, curves[label], strict=True))
    )
    print(
        f"  next-second {ETF} move: own last move R^2 {own:.4f}, "
        f"plus the basis (EWMA {TAU} s) {both:.4f}"
    )

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
fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.0), facecolor=SURFACE)
for a in (left, right):
    a.set_facecolor(SURFACE)
    a.grid(color=GRID, lw=0.6)
    a.set_axisbelow(True)
left.scatter(ticks, Chy[0, 1:] / base[0, 1:], s=14, color=ORANGE, label="Hayashi-Yoshida")
left.scatter(ticks, P[1.0][0, 1:] / base[0, 1:], s=14, color=BLUE, label="pre-averaged HY")
left.axhline(1.0, color=MUTED, lw=0.8, ls="--")
left.set_xscale("log")
left.set_xlabel("top-of-book updates, 10:00-13:00")
left.set_ylabel("covariance with QQQ / 60 s estimate")
left.set_title("covariance from ticks, per component", color=INK, fontsize=9.5)
left.legend(frameon=False, fontsize=8)
for (label, v), col in zip(curves.items(), (BLUE, GREEN), strict=True):
    right.plot(KS, v, "o-", color=col, lw=1.4, ms=3.5, label=label)
right.plot(KS, 1 / KS, color=MUTED, lw=0.8, ls="--", label="pure noise, 1/k")
right.axhline(1.0, color=MUTED, lw=0.8)
right.set_xscale("log")
right.set_yscale("log")
right.set_xlabel("horizon k (seconds)")
right.set_ylabel("variance ratio of QQQ minus basket")
right.set_title("the unhedged part reverts", color=INK, fontsize=9.5)
right.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "docs" / "etf_basis.png", dpi=150, facecolor=SURFACE)
