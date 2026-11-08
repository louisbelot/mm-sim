import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

from mmsim.itch import read
from mmsim.micro import Tops, tops
from mmsim.ofi import bins, oos_r2, regress, scaling, windows

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1", "#d4a017", "#c2185b"]
ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = sys.argv[1:] or ["CSCO", "INTC", "CMCSA", "MSFT", "AAPL", "QQQ"]
OPEN, NOON, CLOSE = 575.0, 755.0, 955.0
WINDOW, DT = 30.0, 10 / 60


def load(symbol: str) -> Tops:
    path = ROOT / "data" / f"{symbol}.tops.npz"
    if not path.exists():
        np.savez(path, **tops(read(ROOT / "data" / f"{symbol}.itch"))._asdict())
    z = np.load(path)
    return Tops(*(z[k] for k in Tops._fields))


def study(symbol: str):
    top = load(symbol)
    ws = windows(top, OPEN, CLOSE, WINDOW, DT)
    am, pm = bins(top, OPEN, NOON, DT), bins(top, NOON, CLOSE, DT)
    now = regress(am.ofi, am.dp)
    scaled = regress(am.ofi / am.depth, am.dp)
    ahead = regress(am.ofi[:-1], am.dp[1:])
    return {
        "windows": ws,
        "r2": np.mean([w.fit.r2 for w in ws]),
        "t": np.median([w.fit.t for w in ws]),
        "c": np.mean([w.fit.beta * w.depth for w in ws]),
        "lam": scaling(ws),
        "oos": oos_r2(pm.dp, now.predict(pm.ofi)),
        "scaled": oos_r2(pm.dp, scaled.predict(pm.ofi / pm.depth)),
        "ahead": oos_r2(pm.dp[1:], ahead.predict(pm.ofi[:-1])),
        "example": bins(top, 720.0, 750.0, DT),
    }


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def chart(out, pooled, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
    style(a)
    first = SYMBOLS[0]
    ex = out[first]["example"]
    f = regress(ex.ofi, ex.dp)
    a.scatter(ex.ofi / 1000, ex.dp, s=12, color=COLORS[0], alpha=0.6, edgecolor="none")
    x = np.linspace(ex.ofi.min(), ex.ofi.max(), 2)
    a.plot(
        x / 1000,
        f.predict(x),
        color=INK,
        linewidth=1.4,
        label=f"slope {f.beta * 1000:.2f} ticks per 1000 shares, R$^2$ {f.r2:.2f}",
    )
    a.set_xlabel(f"{first} order flow imbalance in 10 s, thousands of shares", color=INK)
    a.set_ylabel("mid change in 10 s, ticks (12:00 to 12:30)", color=INK)
    a.legend(frameon=False, labelcolor=INK, fontsize=8.5, loc="upper left")

    style(b)
    for c, s in zip(COLORS, SYMBOLS, strict=False):
        ws = out[s]["windows"]
        b.scatter([w.depth for w in ws], [w.fit.beta for w in ws], s=18, color=c, label=s)
    d = np.array([w.depth for s in SYMBOLS for w in out[s]["windows"]])
    x = np.geomspace(d.min(), d.max(), 2)
    b.plot(
        x,
        np.exp(pooled.alpha) * x**pooled.beta,
        color=INK,
        linewidth=1.4,
        label=f"fit, slope {pooled.beta:.2f}",
    )
    b.plot(x, 0.5 / x, color=MUTED, linestyle="--", linewidth=1.2, label="1 / (2 depth)")
    b.set_xscale("log")
    b.set_yscale("log")
    b.xaxis.set_minor_formatter(NullFormatter())
    b.set_xticks([200, 500, 1000, 2000, 5000], ["200", "500", "1000", "2000", "5000"])
    b.set_xlabel("mean depth at the best quotes, shares", color=INK)
    b.set_ylabel("price impact coefficient, ticks per share", color=INK)
    b.legend(frameon=False, labelcolor=INK, fontsize=8.5, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    out = {s: study(s) for s in SYMBOLS}
    print("10 s intervals, regressions in 30-minute windows, 09:35 to 15:35")
    print(
        "symbol  mean R^2  median t  beta*depth  log-slope (se)  pm R^2: am fit  /depth  next 10 s"
    )
    for s, r in out.items():
        lam = r["lam"]
        print(
            f"{s:6s}  {r['r2']:8.2f}  {r['t']:8.1f}  {r['c']:10.2f}  {lam.beta:6.2f} "
            f"({lam.beta / lam.t:4.2f})  {r['oos']:14.2f}  {r['scaled']:6.2f}  {r['ahead']:9.4f}"
        )
    pooled = scaling([w for r in out.values() for w in r["windows"]])
    print(
        f"pooled log-slope {pooled.beta:.3f} (se {pooled.beta / pooled.t:.3f}), {pooled.n} windows"
    )
    chart(out, pooled, ROOT / "docs" / "ofi.png")
