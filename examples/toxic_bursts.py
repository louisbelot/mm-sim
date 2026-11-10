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

from mmsim.bots import GLFT, RegimeLean, Widen
from mmsim.detect import one_sided, toxic
from mmsim.flow import CSTParams, ZeroIntelligence
from mmsim.glft import Model
from mmsim.informed import Bursty, Fundamental
from mmsim.sim import Agent, Simulator
from mmsim.tournament import session

SKY = CSTParams.sky_perfect()
RATE, ON, OFF = 2.0, 1 / 40, 1 / 10
MINUTES, WARM = 120.0, 30.0
MODEL = Model(SIGMA, A, K, 0.08, Q)
SHIFTS = (0.25, 0.5, 1.0, 2.0)
WIDTHS = (1.0, 2.0, 3.0)
TUNE = range(10_000_000, 10_000_300)
TEST = range(1000)
STEP = 0.25
SHOW = 50


def flow():
    return ZeroIntelligence(SKY)


def informed(seed):
    return Bursty(Fundamental(1.0, seed=seed), RATE, ON, OFF, seed=seed)


def lean(shift):
    return RegimeLean(one_sided(SKY.mu, RATE, ON, OFF), shift)


def widen(ticks):
    return Widen(toxic(SKY.mu, RATE, ON, OFF), ticks)


def maker(name, shift, ticks):
    match name:
        case "GLFT":
            return lambda: GLFT(MODEL, keep=1)
        case "lean":
            return lambda: GLFT(MODEL, keep=1, signal=lean(shift))
        case "widen":
            return lambda: GLFT(MODEL, keep=1, widen=widen(ticks))
    return lambda: GLFT(MODEL, keep=1, signal=lean(shift), widen=widen(ticks))


NAMES = ("GLFT", "lean", "widen", "lean + widen")


def play(args):
    name, shift, ticks, seed = args
    r = session(
        {"m": maker(name, shift, ticks)},
        seed,
        flow=flow,
        minutes=MINUTES,
        crowd=lambda s: [informed(s)],
    )["m"]
    return r.pnl - PHI * r.exposure - TERM * r.position**2, r.pnl, r.spread, r.adverse, r.inventory


class Watch(Agent):
    def __init__(self):
        self.rate = toxic(SKY.mu, RATE, ON, OFF)
        self.side = one_sided(SKY.mu, RATE, ON, OFF)
        self.path = []

    def on_start(self, ctx):
        self.rate.t = self.side.t = ctx.t
        ctx.set_timer(0.0)

    def on_trade(self, ctx, p):
        self.rate.observe(p.t, 0, p.qty)
        self.side.observe(p.t, p.side, p.qty)

    def on_timer(self, ctx):
        q = self.side.at(ctx.t)
        self.path.append((self.rate.at(ctx.t)[1], q[1] + q[2]))
        ctx.set_timer(STEP)


def watch(seed):
    w, bot = Watch(), informed(seed)
    sim = Simulator(flow(), [w, bot], seed=seed)
    sim.seed(10_000, 10_001)
    sim.warm_up(WARM)
    sim.run(MINUTES)
    grid = WARM + np.arange(len(w.path)) * STEP
    on = np.zeros(len(grid), dtype=bool)
    flips = bot.switches + [np.inf] * (len(bot.switches) % 2)
    for a, b in zip(flips[::2], flips[1::2], strict=True):
        on |= (grid >= a) & (grid < b)
    trace = None
    if seed < SHOW:
        trace = (seed, sim.tape.t, sim.tape.mid, bot.value.path, flips)
    return np.array(w.path), on, flips, trace


def delays(p, on, flips):
    grid = WARM + np.arange(len(p)) * STEP
    out = []
    for a, b in zip(flips[::2], flips[1::2], strict=True):
        if a >= grid[-1]:
            continue
        inside = (grid >= a) & (grid < min(b, grid[-1]))
        hit = np.flatnonzero(p[inside] > 0.5)
        out.append(hit[0] * STEP if len(hit) else np.inf)
    return out


def mean_se(x, axis=0):
    x = np.asarray(x, dtype=float)
    return x.mean(axis), x.std(axis, ddof=1) / np.sqrt(x.shape[axis])


def chart(trace, posterior, waits, table, path):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4), facecolor=SURFACE)
    for ax in axes.flat:
        style(ax)
    (a, b), (c, d) = axes

    seed, t, mid, value, flips = trace
    t = np.asarray(t) - WARM
    keep = t >= 0
    a.step(t[keep], np.asarray(mid)[keep], where="post", color=MUTED, linewidth=1, label="mid")
    vt, vv = zip(*value, strict=True)
    a.plot(
        np.asarray(vt) - WARM,
        vv,
        color=ORANGE,
        linewidth=1.2,
        label="fundamental, where the informed trader looked",
    )
    for lo, hi in zip(flips[::2], flips[1::2], strict=True):
        a.axvspan(lo - WARM, min(hi, WARM + MINUTES) - WARM, color=ORANGE, alpha=0.1, lw=0)
    a.set_xlim(0, MINUTES)
    a.yaxis.set_major_formatter("{x:.0f}")
    a.set_xlabel("minute (informed trader active in the shaded bursts)", color=INK)
    a.set_ylabel("price (ticks)", color=INK)
    a.set_title(f"seed {seed}, the most burst time of the first {SHOW}", color=INK, loc="left")
    a.legend(frameon=False, labelcolor=INK, loc="best")

    g = np.arange(len(posterior)) * STEP
    for lo, hi in zip(flips[::2], flips[1::2], strict=True):
        b.axvspan(lo - WARM, min(hi, WARM + MINUTES) - WARM, color=ORANGE, alpha=0.1, lw=0)
    b.plot(g, posterior[:, 0], color=BLUE, linewidth=1.3, label="print rate filter")
    b.plot(g, posterior[:, 1], color=GREEN, linewidth=1.3, label="side filter")
    b.set_xlim(0, MINUTES)
    b.set_ylim(0, 1.18)
    b.set_yticks(np.linspace(0, 1, 6))
    b.set_xlabel("minute", color=INK)
    b.set_ylabel("posterior probability of a burst", color=INK)
    b.set_title("what the two filters see in that session", color=INK, loc="left")
    b.legend(frameon=False, labelcolor=INK, loc="upper center", ncols=2)

    horizon = np.linspace(0, 10, 81)
    for (name, w), color in zip(waits.items(), (BLUE, GREEN), strict=True):
        w = np.asarray(w)
        c.plot(horizon, [(w <= h).mean() for h in horizon], color=color, linewidth=1.5, label=name)
    c.set_ylim(0, 1)
    c.set_xlabel("minutes since the burst started", color=INK)
    c.set_ylabel("share of bursts flagged (posterior over 0.5)", color=INK)
    c.set_title("how fast each filter flags a burst", color=INK, loc="left")
    c.legend(frameon=False, labelcolor=INK, loc="lower right")

    pos = np.arange(len(NAMES))
    w = 0.26
    parts = (("score", 0, INK), ("spread", 2, BLUE), ("adverse", 3, ORANGE))
    for j, (label, k, color) in enumerate(parts):
        m = [table[n][0][k] for n in NAMES]
        e = [2 * table[n][1][k] for n in NAMES]
        d.bar(pos + (j - 1) * w, m, w, yerr=e, color=color, label=label, ecolor=MUTED)
    d.axhline(0, color=MUTED, linewidth=0.8)
    d.set_ylim(None, 1.3 * max(table[n][0][2] for n in NAMES))
    d.set_xticks(pos, NAMES)
    d.set_ylabel("ticks per session", color=INK)
    d.set_title(f"GLFT makers, {len(TEST)} sessions of {MINUTES:g} minutes", color=INK, loc="left")
    d.legend(frameon=False, labelcolor=INK, loc="upper center", ncols=3)

    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    t0 = time.perf_counter()
    with ProcessPoolExecutor() as pool:
        runs = list(pool.map(watch, TEST, chunksize=20))
        waits = {"print rate filter": [], "side filter": []}
        alarms = np.zeros(2)
        quiet = 0
        for p, on, flips, _ in runs:
            for j, name in enumerate(waits):
                waits[name] += delays(p[:, j], on, flips)
            alarms += (p[~on] > 0.5).sum(0)
            quiet += (~on).sum()
        bursts = len(waits["side filter"])
        print(f"{bursts} bursts in {len(TEST)} sessions of {MINUTES:g} minutes")
        for j, (name, w) in enumerate(waits.items()):
            w = np.asarray(w)
            print(
                f"{name:18s} flagged {np.isfinite(w).mean():.0%}, within 2 minutes "
                f"{(w <= 2).mean():.0%}, median delay {np.median(w[np.isfinite(w)]):.2f}"
                f", false alarms {alarms[j] / quiet:.1%} of quiet time"
            )
        posterior, on, _, trace = max(
            (r for r in runs if r[3] is not None), key=lambda r: r[1].mean()
        )

        def tune(name, grid, fixed):
            out = {}
            for x in grid:
                args = (x, fixed) if name == "lean" else (fixed, x)
                rows = pool.map(play, [(name, *args, s) for s in TUNE], chunksize=10)
                out[x] = mean_se([r[0] for r in rows])
            for x, (m, se) in out.items():
                print(f"  {name} {x:g}: {m:.2f} ({se:.2f})")
            return max(out, key=lambda x: out[x][0])

        print(f"\ntuning on {len(TUNE)} sessions")
        shift = tune("lean", SHIFTS, 0.0)
        ticks = tune("widen", WIDTHS, 0.0)
        print(f"shift {shift:g}, widen {ticks:g} ticks at certainty")

        print(f"\n{len(TEST)} held-out sessions, mean (se)")
        print("maker            score          pnl  spread  adverse  inventory")
        table = {}
        for name in NAMES:
            rows = list(pool.map(play, [(name, shift, ticks, s) for s in TEST], chunksize=10))
            table[name] = m, se = mean_se(rows)
            print(
                f"{name:14s} {m[0]:7.2f} ({se[0]:.2f})  {m[1]:6.2f}  {m[2]:6.2f}"
                f"  {m[3]:7.2f}  {m[4]:9.2f}"
            )

    chart(trace, posterior, waits, table, ROOT / "docs" / "toxic_bursts.png")
    print(f"\ndone in {time.perf_counter() - t0:.0f} s")
