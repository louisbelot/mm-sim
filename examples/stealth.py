import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from rl_market_maker import (
    BLUE,
    GREEN,
    INK,
    MUTED,
    ORANGE,
    PHI,
    ROOT,
    SIGMA,
    SURFACE,
    TERM,
    A,
    K,
    Q,
    style,
)

from mmsim.book import Side
from mmsim.bots import GLFT, RegimeLean
from mmsim.detect import one_sided
from mmsim.execution import Executor, Passive, Pov, fit_response, impulse, scatter, shortfall, trial
from mmsim.flow import CSTParams, Resilient, ZeroIntelligence
from mmsim.glft import Model
from mmsim.sim import Agent, Simulator

SKY = CSTParams.sky_perfect()
C = 0.5
LAGS = np.array([0, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32])
FIT = {"plain": LAGS >= 0, "resilient": LAGS >= 1}
X, START, LENGTH, MINUTES = 20, 20.0, 20.0, 60.0
SHIFT = 1.0
RATE = 0.4
STYLES = ("even", "random", f"POV {RATE:.0%}", "passive")
MODEL = Model(SIGMA, A, K, 0.08, Q)
RUNS = range(1000)


def plain():
    return ZeroIntelligence(SKY)


def resilient():
    return Resilient(SKY, C, SIGMA)


BOOKS = {"plain": plain, "resilient": resilient}


def gap(seed):
    f = resilient()
    sim = Simulator(f, seed=seed, record_every=None)
    sim.seed(10_000, 10_001)
    sim.warm_up(30.0)
    out = []
    for _ in range(800):
        sim.run(0.25)
        out.append(sim.tape.at(sim.t) - f.value)
    return out


def respond(args):
    book, seed = args
    return impulse(BOOKS[book], 2, LAGS, seed)


def executor(style, seed):
    side = Side.BUY if seed % 2 == 0 else Side.SELL
    match style:
        case "even":
            return Executor(side, START + LENGTH * (np.arange(X) + 0.5) / X, np.arange(1, X + 1))
        case "random":
            times, done = scatter(X, LENGTH, X // 2, np.random.default_rng(seed))
            return Executor(side, START + times, done)
        case "passive":
            k = np.arange(11)
            return Passive(side, START + LENGTH * k / 10, X * k // 10, slack=2)
    return Pov(side, X, RATE, START)


def filt():
    return one_sided(SKY.mu, 1.0, 1 / MINUTES, 1 / LENGTH)


class Watch(Agent):
    def __init__(self, side):
        self.filter = filt()
        self.side = side
        self.path = []

    def on_start(self, ctx):
        self.filter.t = ctx.t
        ctx.set_timer(0.0)

    def on_trade(self, ctx, p):
        self.filter.observe(p.t, p.side, p.qty)

    def on_timer(self, ctx):
        self.path.append(self.filter.at(ctx.t)[1 + self.side])
        ctx.set_timer(0.25)


def alone(args):
    book, style, seed = args
    bot = executor(style, seed)
    w = Watch(bot.side)
    sim = Simulator(BOOKS[book](), [w, bot], seed=seed, record_every=None)
    sim.seed(10_000, 10_001)
    sim.warm_up(30.0)
    t0 = sim.t
    sim.run(MINUTES)
    fills = sim.contexts[1].account.fills
    sign = 1 if bot.side == Side.BUY else -1
    end = sign * (sim.tape.at(sim.t) - bot.arrival)
    finish = max(f.t for f in fills) - t0 - START
    taken = sum(f.qty for f in fills if not f.maker)
    full = abs(sim.contexts[1].position) == X
    return w.path, shortfall(fills, bot.arrival) / X, finish, taken, end, full


def maker(lean):
    if lean:
        return {"m": lambda: GLFT(MODEL, keep=1, signal=RegimeLean(filt(), SHIFT))}
    return {"m": lambda: GLFT(MODEL, keep=1)}


def meet(args):
    book, style, lean, seed = args
    bot = None if style is None else executor(style, seed)
    res, _ = trial(seed, maker(lean), bot, flow=BOOKS[book], minutes=MINUTES)
    r = res["m"]
    return r.pnl - PHI * r.exposure - TERM * r.position**2


def mean_se(x, axis=0):
    x = np.asarray(x, dtype=float)
    return x.mean(axis), x.std(axis, ddof=1) / np.sqrt(x.shape[axis])


def chart(resp, fits, paths, costs, lean, path):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4), facecolor=SURFACE)
    for ax in axes.flat:
        style(ax)
    (a, b), (c, d) = axes
    colors = dict(zip(STYLES, (MUTED, BLUE, GREEN, ORANGE), strict=True))

    x = np.arange(len(LAGS))
    for (book, (m, se)), color in zip(resp.items(), (MUTED, ORANGE), strict=True):
        a.errorbar(x, m[:, 1], 2 * se[:, 1], color=color, marker="o", markersize=4, label=book)
        a.plot(
            x[FIT[book]],
            fits[book].response(LAGS[FIT[book]]),
            color=color,
            linestyle="--",
            linewidth=1,
        )
    a.axhline(0, color=MUTED, linewidth=0.8)
    a.set_xticks(x, [f"{lag:g}" for lag in LAGS])
    a.set_xlabel("minutes since a market buy", color=INK)
    a.set_ylabel("ask move per share bought (ticks)", color=INK)
    a.set_title("impact that stays and impact that decays", color=INK, loc="left")
    a.legend(frameon=False, labelcolor=INK, loc="upper right")

    t = np.arange(len(paths["even"])) * 0.25
    b.axvspan(START, START + LENGTH, color=ORANGE, alpha=0.08, linewidth=0)
    for s, p in paths.items():
        b.plot(t, p, color=colors[s], linewidth=1.4, label=s)
    b.set_ylim(0, 1)
    b.set_xlabel("minute of the session (executor works in the shaded window)", color=INK)
    b.set_ylabel("mean posterior on the executor's side", color=INK)
    b.set_title(f"the flow filter on {X} shares, plain book", color=INK, loc="left")
    b.legend(frameon=False, labelcolor=INK, loc="upper right")

    w = 0.38
    pos = np.arange(len(STYLES))
    for j, (book, color) in enumerate((("plain", MUTED), ("resilient", ORANGE))):
        m = [costs[book][s][0] for s in STYLES]
        e = [2 * costs[book][s][1] for s in STYLES]
        c.bar(pos + (j - 0.5) * w, m, w, yerr=e, color=color, label=book, ecolor=INK)
    c.axhline(0, color=MUTED, linewidth=0.8)
    c.set_xticks(pos, STYLES)
    c.set_ylabel("shortfall per share vs arrival mid (ticks)", color=INK)
    c.set_ylim(0, 1.3 * max(costs["plain"][s][0] for s in STYLES))
    c.set_title("what the executor pays, no market maker", color=INK, loc="left")
    c.legend(frameon=False, labelcolor=INK, loc="upper left", ncols=2)

    labels = ["none", *STYLES]
    pos = np.arange(len(labels))
    for j, (book, color) in enumerate((("plain", MUTED), ("resilient", ORANGE))):
        m = [lean[book][s][0] for s in labels]
        e = [2 * lean[book][s][1] for s in labels]
        d.bar(pos + (j - 0.5) * w, m, w, yerr=e, color=color, label=book, ecolor=INK)
    d.axhline(0, color=MUTED, linewidth=0.8)
    d.set_xticks(pos, labels)
    d.set_ylabel("score with the filter minus without (ticks)", color=INK)
    d.set_title("does leaning with the flow pay a GLFT maker", color=INK, loc="left")
    d.legend(frameon=False, labelcolor=INK, loc="upper left")

    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    t0 = time.perf_counter()
    with ProcessPoolExecutor() as pool:
        print("ask move per share after a 2-share market buy, 2000 paired runs")
        print("book       " + "".join(f"{lag:>7g}" for lag in LAGS))
        resp, fits = {}, {}
        for book in BOOKS:
            d = np.array(list(pool.map(respond, [(book, s) for s in range(2000)], chunksize=25)))
            d = d[~np.isnan(d).any(axis=(1, 2))] / 2
            resp[book] = mean_se(d)
            m, se = resp[book]
            k = FIT[book]
            fits[book] = fit_response(LAGS[k], m[k, 1], se[k, 1])
            f = fits[book]
            print(f"{book:10s} " + "".join(f"{v:7.3f}" for v in m[:, 1]))
            print(
                f"{'':10s} fit from minute {LAGS[k][0]:g}: permanent {f.lam:.3f}, "
                f"decaying {f.kappa:.3f}, "
                f"rho {f.rho:.3f} (half-life {np.log(2) / f.rho:.2f})"
            )
        g = np.array(list(pool.map(gap, range(64))))
        slope = np.mean(1 / np.cosh(C * g) ** 2)
        rate = 2 * C * SKY.mu * fits["plain"].lam
        print(
            f"gap sd {g.std():.2f}, E sech^2(c g) {slope:.3f}, decay rate linearized "
            f"{rate:.3f}, averaged over the gap {rate * slope:.3f}"
        )

        print(f"\n{X} shares from minute {START:g}, no market maker, {len(RUNS)} sessions each")
        print("book       style      cost (se)      taken  done  finish  end impact  found  delay")
        paths, costs = {}, {}
        for book in BOOKS:
            costs[book] = {}
            for s in STYLES:
                out = list(pool.map(alone, [(book, s, i) for i in RUNS], chunksize=20))
                p = np.array([o[0] for o in out])
                cost, finish, taken, end, full = np.array([o[1:] for o in out]).T
                costs[book][s] = m, se = mean_se(cost)
                tt = np.arange(p.shape[1]) * 0.25
                inside = (tt >= START) & (tt < START + LENGTH)
                hit = p[:, inside] > 0.5
                delay = np.where(hit.any(1), hit.argmax(1) * 0.25, np.nan)
                if book == "plain":
                    paths[s] = p.mean(0)
                print(
                    f"{book:10s} {s:10s} {m:6.3f} ({se:.3f})  {taken.mean():5.1f}"
                    f"  {full.mean():4.0%}  {finish.mean():6.1f}  {end.mean():10.2f}"
                    f"  {hit.any(1).mean():5.0%}  {np.nanmedian(delay):5.1f}"
                )
                if s == "even":
                    early = (p[:, tt < START] > 0.5).mean()
            print(f"{book:10s} before the window the posterior is over 0.5 {early:.1%} of the time")

        print(f"\nGLFT maker, {len(RUNS)} sessions per cell, score = pnl - exposure - terminal")
        print("book       style       GLFT (se)       + filter (se)   difference (se)")
        lean = {}
        for book in BOOKS:
            lean[book] = {}
            for s in (None, *STYLES):
                rows = [
                    list(pool.map(meet, [(book, s, f, i) for i in RUNS], chunksize=20))
                    for f in (False, True)
                ]
                g, h = (mean_se(r) for r in rows)
                lean[book][s or "none"] = diff = mean_se(np.subtract(rows[1], rows[0]))
                print(
                    f"{book:10s} {s or 'none':10s} {g[0]:6.2f} ({g[1]:.2f})"
                    f"   {h[0]:6.2f} ({h[1]:.2f})   {diff[0]:6.2f} ({diff[1]:.2f})"
                )

    chart(resp, fits, paths, costs, lean, ROOT / "docs" / "stealth.png")
    print(f"\ndone in {time.perf_counter() - t0:.0f} s")
