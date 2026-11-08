import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmsim.itch import read
from mmsim.micro import Tops, bucket, fit, spreads, tops
from mmsim.ofi import regress

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1", "#d4a017", "#c2185b"]
ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = sys.argv[1:] or ["CSCO", "INTC", "CMCSA", "MSFT", "AAPL", "QQQ"]
OPEN, NOON, CLOSE = 575.0, 755.0, 955.0
HALVES = ((OPEN, NOON), (NOON, CLOSE))
SECOND = 1 / 60
MOVES = (1, 2, 5, 20)
CHART = (1, 2, 3, 5)


def load(symbol: str) -> Tops:
    path = ROOT / "data" / f"{symbol}.tops.npz"
    if not path.exists():
        np.savez(path, **tops(read(ROOT / "data" / f"{symbol}.itch"))._asdict())
    z = np.load(path)
    return Tops(*(z[k] for k in Tops._fields))


def after(top: Tops, i: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    moves = np.flatnonzero(np.diff(top.mid) != 0) + 1
    j = np.searchsorted(moves, i, side="right") + k - 1
    ok = j < len(moves)
    return ok, top.mid[moves[np.minimum(j, len(moves) - 1)]] - top.mid[i]


def grid(lo: float, hi: float) -> np.ndarray:
    return np.arange(lo + SECOND, hi - SECOND, SECOND)


def study(symbol: str):
    top = load(symbol)
    ns = spreads(top.between(OPEN, CLOSE), CLOSE)
    fits = [fit(top.between(*h), n_spread=ns) for h in HALVES]
    sse = {key: np.zeros(2) for key in ("10 s", "2 moves")}
    names = ("weighted mid", "microprice", "imbalance fit")
    err = {(h, n): 0.0 for h in sse for n in names}
    tot = dict.fromkeys(sse, 0.0)
    slopes = {k: ([], [], []) for k in (*sorted({*MOVES, *CHART}), "G1")}
    for train, test in ((0, 1), (1, 0)):
        m = fits[train]

        def features(lo, hi, horizon, m=m):
            t = grid(lo, hi)
            i = top.index(t)
            if horizon == "10 s":
                y = top.mid_at(t + 10 * SECOND) - top.mid[i]
                ok = t + 10 * SECOND < hi
            else:
                ok, y = after(top, i, 2)
            s = top.spread[i]
            g = np.nan_to_num(m.adjust(s, top.bq[i], top.aq[i]))
            imb = top.imbalance[i] - 0.5
            return y[ok], (imb * s)[ok], g[ok], imb[ok]

        for h in sse:
            y0, _, _, x0 = features(*HALVES[train], h)
            y, w, g, x = features(*HALVES[test], h)
            line = regress(x0, y0)
            for n, p in zip(names, (w, g, line.predict(x)), strict=True):
                err[h, n] += float(((y - p) ** 2).sum())
            tot[h] += float((y**2).sum())

        t = grid(*HALVES[test])
        i = top.index(t)
        g = m.adjust(top.spread[i], top.bq[i], top.aq[i])
        s = top.spread[i]
        g1 = m.G1[np.clip(s - 1, 0, ns - 1), bucket(top.imbalance[i], m.n_imb)]
        block = np.floor((t - OPEN) / 30)
        for k in slopes:
            ok, y = after(top, i, 1 if k == "G1" else k)
            x = g1 if k == "G1" else g
            ok &= np.isfinite(x)
            slopes[k][0].append(x[ok])
            slopes[k][1].append(y[ok])
            slopes[k][2].append(block[ok])
    r2 = {(h, n): 1 - err[h, n] / tot[h] for h, n in err}
    calib = {}
    for k, (x, y, b) in slopes.items():
        f = regress(np.concatenate(x), np.concatenate(y), groups=np.concatenate(b))
        calib[k] = (f.beta, f.beta / f.t)
    return symbol, ns, fits[0], r2, calib


def report(symbol, m):
    print(f"{symbol}, microprice fitted 09:35 to 12:35, ticks above the mid")
    centres = (np.arange(m.n_imb) + 0.5) / m.n_imb
    print("imbalance  " + " ".join(f"{c:6.2f}" for c in centres))
    for s in range(m.n_spread):
        print(f"g* s={s + 1}     " + " ".join(f"{v:6.3f}" for v in m.g[s]))
    print("G1 s=1     " + " ".join(f"{v:6.3f}" for v in m.G1[0]))
    print(f"rows {m.counts.sum():,}, spread 1 {m.counts[0].sum() / m.counts.sum():.1%}")


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def chart(out, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
    style(a)
    _, _, m, _, _ = out[0]
    x = (np.arange(m.n_imb) + 0.5) / m.n_imb
    a.plot(x, x - 0.5, color=MUTED, linestyle="--", linewidth=1.2, label="weighted mid, spread 1")
    a.plot(x, m.G1[0], color=COLORS[2], marker="o", ms=4, label="next move only (G1)")
    a.plot(x, m.g[0], color=COLORS[0], marker="o", ms=4, label="microprice, spread 1")
    a.plot(x, m.g[1], color=COLORS[1], marker="o", ms=4, label="microprice, spread 2")
    a.set_xlabel("bid size / (bid size + ask size)", color=INK)
    a.set_ylabel(f"{out[0][0]}: ticks above the mid", color=INK)
    a.legend(frameon=False, labelcolor=INK, fontsize=8.5, loc="upper left")

    style(b)
    ks = np.arange(len(CHART))
    for j, (c, (symbol, _, _, _, calib)) in enumerate(zip(COLORS, out, strict=False)):
        b.errorbar(
            ks + (j - (len(out) - 1) / 2) * 0.06,
            [calib[k][0] for k in CHART],
            yerr=[2 * calib[k][1] for k in CHART],
            color=c,
            marker="o",
            ms=4,
            capsize=3,
            label=symbol,
        )
    b.axhline(1, color=INK, linestyle="--", linewidth=1.2)
    b.set_xticks(ks, [str(k) for k in CHART], color=INK)
    b.set_xlabel("mid moves ahead", color=INK)
    b.set_ylabel("slope of realized change on microprice, 2 se", color=INK)
    b.legend(frameon=False, labelcolor=INK, fontsize=8.5, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    with ProcessPoolExecutor() as pool:
        out = list(pool.map(study, SYMBOLS))
    report(out[0][0], out[0][2])
    print("\nout-of-sample R^2 against no change, fit on one half of the day, tested on the other")
    print("symbol  spreads  | 10 s: wmid   micro  linear | 2 moves: wmid  micro  linear")
    for symbol, ns, _, r2, _ in out:
        row = [
            r2[h, n]
            for h in ("10 s", "2 moves")
            for n in ("weighted mid", "microprice", "imbalance fit")
        ]
        print(f"{symbol:6s}  {ns:7d}  | " + "  ".join(f"{v:6.3f}" for v in row))
    print("\nslope of the change after k mid moves on the microprice (se, 30-minute clusters)")
    print("symbol  " + "  ".join(f"k={k:<11d}" for k in MOVES) + "  G1 at k=1")
    for symbol, _, _, _, calib in out:
        cells = [f"{calib[k][0]:5.2f} ({calib[k][1]:.2f})" for k in (*MOVES, "G1")]
        print(f"{symbol:6s}  " + "  ".join(f"{c:13s}" for c in cells))
    chart(out, ROOT / "docs" / "microprice.png")
