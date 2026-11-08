import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
from micro_quotes import CLOSE, NOON, OPEN, SESSION, STARTS, load, models, msgs, style, summarize

from mmsim.alpha import Model, Solution, decay, fill_fit, implied_eta, responses, solve
from mmsim.attribution import attribute
from mmsim.book import Side
from mmsim.bots import AlphaTouch, SymmetricQuoter
from mmsim.replay import ItchFlow, ReplayBook
from mmsim.sim import Simulator

INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SYMBOLS = sys.argv[1:] or ["CSCO", "INTC", "CMCSA", "MSFT", "AAPL", "QQQ"]
LAGS = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 60]) / 60
DELTA = 1 / 60
Q, N, LOT, PHI, T = 10, 100, 100, 0.01, 20.0
NAMES = ("mid", "touch", "micro 0.0", "cjp flat", "cjp alpha")


def flat(width: float) -> Solution:
    on = np.ones((2 * Q + 1, 2 * N + 1), dtype=bool)
    bid, ask = on.copy(), on.copy()
    bid[-1], ask[0] = False, False
    return Solution(np.linspace(-width, width, 2 * N + 1), np.zeros(on.shape), bid, ask)


def run(symbol, bot, start):
    sim = Simulator(ItchFlow(msgs(symbol)), [bot], book=ReplayBook)
    sim.warm_up(start)
    sim.run(SESSION)
    return sim


def row(sim):
    a = sim.contexts[0].account
    att = attribute(a.fills, sim.tape, 1.0, sim.end)
    pnl = a.value(float(sim.tape.at(sim.end)))
    return np.array([pnl, a.volume, att.spread, att.adverse, att.inventory])


def signal(symbol, half):
    top = load(symbol)
    micro = models(symbol)[half]
    lo, hi = (OPEN, NOON) if half == 0 else (NOON, CLOSE)
    t = np.arange(lo, hi - LAGS[-1], 1 / 60)
    i = top.index(t)
    g = micro.adjust(top.spread[i], top.bq[i], top.aq[i])
    ok = ~np.isnan(g)
    c, zeta = decay(LAGS, responses(t[ok], g[ok], top.mid_at, LAGS))
    return micro, c, zeta, zeta * c * g[ok], float(top.spread[i].mean())


def touch(args):
    symbol, start = args
    sim = run(symbol, AlphaTouch(flat(1.0), None, 0.0, size=LOT), start)
    top, fills = load(symbol), sim.contexts[0].account.fills
    t = np.array([f.t for f in fills])
    j = top.index(t)
    before = np.where(top.t[j] == t, j - 1, j)
    after = top.index(t + DELTA)
    sign = np.array([1.0 if f.side == Side.SELL else -1.0 for f in fills])
    w = np.array([f.qty / LOT for f in fills])
    return row(sim), (before, after, sign, w)


def calibrate(symbol, half, tapes):
    micro, c, zeta, a, spread = signal(symbol, half)
    top = load(symbol)
    before, after, sign, w = (np.concatenate([x[k] for x in tapes]) for k in range(4))
    g0, g1 = (micro.adjust(top.spread[i], top.bq[i], top.aq[i]) for i in (before, after))
    ok = ~np.isnan(g0) & ~np.isnan(g1)
    y = sign * (top.mid[after] - top.mid[before])
    a0, a1 = (sign * zeta * c * g for g in (g0, g1))
    impact, eps = fill_fit(y[ok], a0[ok], a1[ok], w[ok], DELTA, zeta)
    lam = w.sum() / (2 * SESSION * len(tapes))
    width = 1.5 * float(np.abs(a).max())
    eta = implied_eta(float(a.var()), zeta, eps, lam)
    m = Model(lam, zeta, eta, eps, width, spread, impact, PHI, 0.0, Q, N)
    return m, c, micro


def policies(m):
    return solve(m, T), solve(replace(m, eps=0.0, eta=0.0), T)


def session(args):
    symbol, name, start, m, c, micro, sols = args
    if name == "mid":
        bot = SymmetricQuoter(0.5, LOT, Q * LOT)
    elif name == "micro 0.0":
        bot = SymmetricQuoter(0.5, LOT, Q * LOT, signal=micro)
    elif name == "cjp flat":
        bot = AlphaTouch(sols[1], None, 0.0, size=LOT)
    else:
        bot = AlphaTouch(sols[0], micro, m.zeta * c, size=LOT)
    return row(run(symbol, bot, start))


def chart(rows, out, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.8), facecolor=SURFACE)
    x = np.arange(len(SYMBOLS))
    w = 0.8 / len(NAMES)
    shades = [MUTED, "#b9b8b4", "#9cc3ef", ORANGE, BLUE]
    for ax in (a, b):
        style(ax)
    for k, (nm, col) in enumerate(zip(NAMES, shades, strict=True)):
        pos = x + (k - (len(NAMES) - 1) / 2) * w
        pnl = [np.array(rows[s, nm])[:, 0] / 100 for s in SYMBOLS]
        se = [2 * p.std(ddof=1) / np.sqrt(len(p)) for p in pnl]
        a.bar(pos, [p.mean() for p in pnl], w, color=col, label=nm, yerr=se, ecolor=INK, capsize=2)
        b.bar(pos, [out[s, nm][0][3] for s in SYMBOLS], w, color=col)
    a.set_yscale("symlog", linthresh=20)
    a.set_yticks([-1000, -100, -20, -10, 0, 10, 20, 100])
    a.set_yticklabels(["-1000", "-100", "-20", "-10", "0", "10", "20", "100"])
    a.set_ylabel("net $ per 30-min session, 2 se (symlog)", color=INK)
    b.set_ylabel("adverse at 1 min, cents per share traded", color=INK)
    for ax in (a, b):
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_xticks(x, SYMBOLS, color=INK)
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, labelcolor=INK, ncol=len(NAMES), loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    from micro_quotes import ROOT

    half = {t: 0 if t < NOON else 1 for t in STARTS}
    with ProcessPoolExecutor() as pool:
        base = {s: list(pool.map(touch, [(s, t) for t in STARTS])) for s in SYMBOLS}
        fits = {
            (s, h): calibrate(s, h, [base[s][k][1] for k, t in enumerate(STARTS) if half[t] == h])
            for s in SYMBOLS
            for h in (0, 1)
        }
        sols = dict(zip(fits, pool.map(policies, [f[0] for f in fits.values()]), strict=True))
        jobs = {}
        for s in SYMBOLS:
            for nm in NAMES:
                if nm == "touch":
                    continue
                args = []
                for t in STARTS:
                    key = (s, 1 - half[t])
                    m, c, micro = fits[key]
                    args.append((s, nm, t, m, c, micro, sols[key]))
                jobs[s, nm] = pool.map(session, args)
        rows = {k: list(v) for k, v in jobs.items()}
    for s in SYMBOLS:
        rows[s, "touch"] = [r for r, _ in base[s]]
    with np.errstate(invalid="ignore"):
        out = {k: summarize(v) for k, v in rows.items()}
    print("calibration (fitted on one half): c, zeta /min, lam /min, impact, eps, eta, spread")
    for (s, h), (m, c, _) in fits.items():
        print(
            f"{s:6s} {'am' if h == 0 else 'pm'}  c {c:5.2f}  zeta {m.zeta:6.1f}  lam {m.lam:5.2f}"
            f"  impact {m.impact:6.3f}  eps {m.eps:7.2f}  eta {m.eta:7.2f}  spread {m.spread:4.2f}"
        )
    print(f"{len(STARTS)} sessions of {SESSION:.0f} minutes, parameters from the other half")
    print("symbol  quotes      pnl (se)        spread  adverse  inventory  $/session  shares")
    for s in SYMBOLS:
        for nm in NAMES:
            per, se, dollars, vol = out[s, nm]
            if vol == 0:
                print(f"{s:6s}  {nm:9s}  no fills")
                continue
            print(
                f"{s:6s}  {nm:9s} {per[0]:6.3f} ({se:5.3f}) {per[2]:8.3f} {per[3]:8.3f}"
                f" {per[4]:9.3f} {dollars:10.1f} {vol:7.0f}"
            )
    chart(rows, out, ROOT / "docs" / "alpha_quotes.png")
