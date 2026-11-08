import math
import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from microprice import (
    COLORS,
    HALVES,
    INK,
    MUTED,
    OPEN,
    ROOT,
    SECOND,
    SURFACE,
    after,
    grid,
    load,
    style,
)

from mmsim.micro import fit, last_move, spreads
from mmsim.ofi import regress

SYMBOLS = sys.argv[1:] or ["CSCO", "INTC", "CMCSA", "MSFT", "AAPL", "QQQ"]
MOVES = (1, 2, 5, 20)
KINDS = {"plain": 0, "just moved": 1, "10 updates": 10, "sticky": math.inf}


def study(symbol: str):
    top = load(symbol)
    ns = spreads(top.between(OPEN, HALVES[1][1]), HALVES[1][1])
    flags = {k: last_move(top, ttl) for k, ttl in KINDS.items() if ttl}
    up = flags["sticky"] - 1.0
    fits = {
        k: [fit(top.between(*h), n_spread=ns, memory=ttl) for h in HALVES]
        for k, ttl in KINDS.items()
    }
    pairs = {(k, m): ([], [], []) for k in KINDS for m in MOVES}
    err = {(k, h): 0.0 for k in (*KINDS, "linear") for h in ("10 s", "2 moves")}
    tot = dict.fromkeys(("10 s", "2 moves"), 0.0)
    for train, test in ((0, 1), (1, 0)):
        lo, hi = HALVES[test]
        t = grid(lo, hi)
        i = top.index(t)
        block = np.floor((t - OPEN) / 30)
        g = {
            k: fits[k][train].adjust(
                top.spread[i], top.bq[i], top.aq[i], flags[k][i] if ttl else None
            )
            for k, ttl in KINDS.items()
        }
        for k in KINDS:
            for m in MOVES:
                ok, y = after(top, i, m)
                ok &= np.isfinite(g[k])
                for lst, v in zip(pairs[k, m], (g[k][ok], y[ok], block[ok]), strict=True):
                    lst.append(v)
        t0 = grid(*HALVES[train])
        i0 = top.index(t0)
        for h in ("10 s", "2 moves"):
            if h == "10 s":
                ok, y = t + 10 * SECOND < hi, top.mid_at(t + 10 * SECOND) - top.mid[i]
                ok0, y0 = (
                    t0 + 10 * SECOND < HALVES[train][1],
                    top.mid_at(t0 + 10 * SECOND) - top.mid[i0],
                )
            else:
                ok, y = after(top, i, 2)
                ok0, y0 = after(top, i0, 2)
            ok &= np.all([np.isfinite(v) for v in g.values()], axis=0)
            X0 = np.column_stack([top.imbalance[i0] - 0.5, up[i0]])[ok0]
            beta = np.linalg.lstsq(X0, y0[ok0], rcond=None)[0]
            X = np.column_stack([top.imbalance[i] - 0.5, up[i]])
            for k, p in (*g.items(), ("linear", X @ beta)):
                err[k, h] += float(((y[ok] - p[ok]) ** 2).sum())
            tot[h] += float((y[ok] ** 2).sum())
    calib = {}
    for key, (x, y, b) in pairs.items():
        f = regress(np.concatenate(x), np.concatenate(y), groups=np.concatenate(b))
        calib[key] = (f.beta, f.beta / f.t)
    r2 = {key: 1 - e / tot[key[1]] for key, e in err.items()}
    return symbol, fits["just moved"][0], calib, r2


def chart(out, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE, sharey=True)
    x = np.arange(len(out))
    w = 0.8 / len(KINDS)
    shades = (MUTED, COLORS[0], "#9cc3ef", COLORS[1])
    for ax, m in zip(axes, (2, 5), strict=True):
        style(ax)
        for j, (k, col) in enumerate(zip(KINDS, shades, strict=True)):
            v = [c[k, m][0] for _, _, c, _ in out]
            e = [2 * c[k, m][1] for _, _, c, _ in out]
            pos = x + (j - (len(KINDS) - 1) / 2) * w
            ax.bar(pos, v, w, color=col, label=k, yerr=e, ecolor=INK, capsize=2)
        ax.axhline(1, color=INK, linewidth=0.8, linestyle="--")
        ax.set_xticks(x, [s for s, *_ in out], color=INK)
        ax.set_title(f"change after {m} mid moves", color=INK, fontsize=10)
    axes[0].set_ylabel("slope of realized change on microprice, 2 se", color=INK)
    axes[0].legend(frameon=False, labelcolor=INK, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    with ProcessPoolExecutor() as pool:
        out = list(pool.map(study, SYMBOLS))
    symbol, m = out[0][0], out[0][1]
    print(f"{symbol}, one-tick spread, g* when the last update moved the mid (ticks), morning fit")
    print("imbalance     " + " ".join(f"{(b + 0.5) / m.n_imb:6.2f}" for b in range(m.n_imb)))
    for flag, name in ((0, "just down"), (1, "no move"), (2, "just up")):
        print(f"{name:12s}  " + " ".join(f"{v:6.3f}" for v in m.g[0, :, flag]))
    print()
    print("slope of the change after k mid moves on g* (se, 30-minute clusters), out of sample")
    print("symbol  model       " + "".join(f"{f'k = {k}':>15s}" for k in MOVES))
    for s, _, c, _ in out:
        for k in KINDS:
            cells = "".join(f"{c[k, m][0]:8.2f} ({c[k, m][1]:.2f})" for m in MOVES)
            print(f"{s:6s}  {k:10s}  {cells}")
    print()
    print("out-of-sample R^2")
    print("symbol  horizon  " + "".join(f"{k:>11s}" for k in (*KINDS, "linear")))
    for s, _, _, r in out:
        for h in ("10 s", "2 moves"):
            print(f"{s:6s}  {h:8s} " + "".join(f"{r[k, h]:11.3f}" for k in (*KINDS, "linear")))
    chart(out, ROOT / "docs" / "micro_memory.png")
