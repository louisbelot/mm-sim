import sys
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmsim.attribution import attribute
from mmsim.bots import SymmetricQuoter
from mmsim.itch import read
from mmsim.micro import Microprice, Tops, fit, spreads, tops
from mmsim.replay import ItchFlow, ReplayBook
from mmsim.sim import Simulator

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = sys.argv[1:] or ["CSCO", "INTC", "CMCSA", "MSFT", "AAPL", "QQQ"]
OPEN, NOON, CLOSE, SESSION = 575.0, 755.0, 955.0, 30.0
STARTS = [OPEN + SESSION * k for k in range(int((CLOSE - OPEN) // SESSION))]
CUTS = (None, 0.0, 0.1, 0.2, 0.3)


@cache
def msgs(symbol):
    return read(ROOT / "data" / f"{symbol}.itch")


def load(symbol: str) -> Tops:
    path = ROOT / "data" / f"{symbol}.tops.npz"
    if not path.exists():
        np.savez(path, **tops(msgs(symbol))._asdict())
    z = np.load(path)
    return Tops(*(z[k] for k in Tops._fields))


def name(cut):
    return "mid" if cut is None else f"micro {cut:.1f}"


def session(args):
    symbol, cut, model, start = args
    signal = None if cut is None else model
    bot = SymmetricQuoter(0.5 if cut is None else 0.5 - cut, 100, 1000, signal=signal)
    sim = Simulator(ItchFlow(msgs(symbol)), [bot], book=ReplayBook)
    sim.warm_up(start)
    sim.run(SESSION)
    a = sim.contexts[0].account
    att = attribute(a.fills, sim.tape, 1.0, sim.end)
    pnl = a.value(float(sim.tape.at(sim.end)))
    return np.array([pnl, a.volume, att.spread, att.adverse, att.inventory])


def summarize(rows):
    rows = np.array(rows)
    pnl, vol = rows[:, 0], rows[:, 1]
    per = rows.sum(axis=0) / vol.sum()
    se = (pnl - per[0] * vol).std(ddof=1) * np.sqrt(len(rows)) / vol.sum()
    return per, se, pnl.mean() / 100, vol.mean()


def models(symbol: str) -> tuple[Microprice, Microprice]:
    top = load(symbol)
    ns = spreads(top.between(OPEN, CLOSE), CLOSE)
    return fit(top.between(OPEN, NOON), n_spread=ns), fit(top.between(NOON, CLOSE), n_spread=ns)


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8, axis="y")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def chart(out, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    x = np.arange(len(SYMBOLS))
    w = 0.8 / len(CUTS)
    shades = [MUTED, "#9cc3ef", "#5f9ee3", BLUE, "#174a87"]
    for ax, j, label in ((a, 3, "adverse, 1 min"), (b, 0, "net, 2 se")):
        style(ax)
        for k, (cut, c) in enumerate(zip(CUTS, shades, strict=True)):
            v = [out[s, cut][0][j] for s in SYMBOLS]
            err = [2 * out[s, cut][1] for s in SYMBOLS] if j == 0 else None
            pos = x + (k - (len(CUTS) - 1) / 2) * w
            ax.bar(pos, v, w, color=c, label=name(cut), yerr=err, ecolor=INK, capsize=2)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_xticks(x, SYMBOLS, color=INK)
        ax.set_ylabel(f"{label}, cents per share traded", color=INK)
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, labelcolor=INK, ncol=len(CUTS), loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    with ProcessPoolExecutor() as pool:
        fits = dict(zip(SYMBOLS, pool.map(models, SYMBOLS), strict=True))
        jobs = {
            (s, cut): pool.map(
                session, [(s, cut, fits[s][1 if t < NOON else 0], t) for t in STARTS]
            )
            for s in SYMBOLS
            for cut in CUTS
        }
        out = {k: summarize(list(v)) for k, v in jobs.items()}
    print(f"{len(STARTS)} sessions of {SESSION:.0f} minutes, microprice fitted on the other half")
    print("symbol  quotes      pnl (se)        spread  adverse  inventory  $/session  shares")
    for (s, cut), (per, se, dollars, vol) in out.items():
        print(
            f"{s:6s}  {name(cut):9s} {per[0]:6.3f} ({se:5.3f}) {per[2]:8.3f} {per[3]:8.3f}"
            f" {per[4]:9.3f} {dollars:10.1f} {vol:7.0f}"
        )
    chart(out, ROOT / "docs" / "micro_quotes.png")
