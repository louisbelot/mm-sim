import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from rl_market_maker import BLUE, GREEN, INK, MUTED, ORANGE, ROOT, SURFACE, style

from mmsim.afs import Profile, cost, optimal
from mmsim.book import Side
from mmsim.execution import Executor, fit_response, impulse, shortfall
from mmsim.flow import CSTParams, ZeroIntelligence
from mmsim.sim import Simulator

SKY = CSTParams.sky_perfect()
LAGS = np.array([0, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8])
LEVELS = 12
SWEEPS = (1, 2, 6, 12)
X = 12
N = 10
HORIZONS = (0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4)
RUNS = 4000
RETRY = 0.05 * np.arange(1, 41)
PLANS = ("block", "even", "OW", "AFS price", "AFS volume")
COLORS = dict(zip(PLANS, (MUTED, INK, BLUE, GREEN, ORANGE), strict=True))


def sky():
    return ZeroIntelligence(SKY)


def respond(args):
    qty, seed = args
    return impulse(sky, qty, LAGS, seed)


def refill(args):
    qty, seed = args
    rows = []
    for n in (qty, 0):
        sim = Simulator(sky(), [Executor(Side.BUY, [0.0], [n])], seed=seed, record_every=None)
        sim.seed(10_000, 10_001)
        sim.warm_up(30.0)
        p0 = sim.book.best_ask()
        row = []
        for dt in np.diff(LAGS, prepend=0.0):
            sim.run(dt)
            row.append(
                sum(q.qty for p in range(p0, p0 + 16) if (q := sim.book.level(Side.SELL, p)))
            )
        rows.append(row)
    return np.subtract(rows[1], rows[0])


def snapshot(seed):
    sim = Simulator(sky(), seed=seed, record_every=None)
    sim.seed(10_000, 10_001)
    sim.warm_up(30.0)
    rows = []
    for _ in range(20):
        sim.run(1.0)
        ask = sim.book.best_ask()
        rows.append(
            [q.qty if (q := sim.book.level(Side.SELL, ask + k)) else 0 for k in range(LEVELS)]
        )
    return rows


def race(args):
    times, done, seed = args
    bot = Executor(Side.BUY, times, done)
    sim = Simulator(sky(), [bot], seed=seed, record_every=None)
    sim.seed(10_000, 10_001)
    sim.warm_up(30.0)
    sim.run(times[-1] + 1.0)
    fills = sim.contexts[-1].account.fills
    return shortfall(fills, bot.arrival), sum(f.qty for f in fills)


def plans(T, ow, rho_e, shape):
    return {
        "block": np.array([X] + [0.0] * N),
        "even": np.full(N + 1, X / (N + 1)),
        "OW": ow.trades(X, T, N),
        "AFS price": optimal(shape, X, N, np.exp(-ow.rho * T / N), "price"),
        "AFS volume": optimal(shape, X, N, np.exp(-rho_e * T / N), "volume"),
    }


def predicted(name, lots, T, ow, rho_e, shape):
    lots = np.asarray(lots, dtype=float)
    match name:
        case "AFS price":
            return cost(shape, lots, np.exp(-ow.rho * T / N), "price")
        case "AFS volume":
            return cost(shape, lots, np.exp(-rho_e * T / N), "volume")
        case _:
            return ow.cost(lots, np.linspace(0.0, T, N + 1))


def done(xi):
    return tuple(int(v) for v in np.rint(np.cumsum(xi)))


def mean_se(x):
    x = np.asarray(x, dtype=float)
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))


def chart(depth, ow, ask, gone, rho_e, table, path):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4), facecolor=SURFACE)
    for ax in axes.flat:
        style(ax)
    (a, b), (c, d) = axes
    a.bar(np.arange(LEVELS), depth, color=BLUE, width=0.7)
    a.axhline(1 / (ow.lam + ow.kappa), color=ORANGE, linestyle="--", linewidth=1.2)
    a.annotate(
        "block with the impulse fit's slope",
        (LEVELS - 0.6, 1 / (ow.lam + ow.kappa)),
        xytext=(0, 6),
        textcoords="offset points",
        ha="right",
        color=ORANGE,
    )
    a.set_xlabel("ticks above the best ask", color=INK)
    a.set_ylabel("mean shares resting", color=INK)
    a.set_title("the shape of the zero-intelligence book", color=INK, loc="left")
    x = np.arange(len(LAGS))
    for (q, m), alpha in zip(gone.items(), (0.4, 0.6, 0.8, 1.0), strict=True):
        b.plot(x, m / m[0], color=ORANGE, alpha=alpha, marker="o", markersize=3, label=f"{q}")
    b.plot(x, np.exp(-rho_e * LAGS), color=ORANGE, linestyle="--", linewidth=1.1)
    b.plot(x, (ask - ow.lam) / ow.kappa, color=BLUE, marker="o", markersize=3)
    b.plot(x, np.exp(-ow.rho * LAGS), color=BLUE, linestyle="--", linewidth=1.1)
    b.annotate(
        "missing ask volume",
        (6, np.exp(-rho_e)),
        xytext=(6, 8),
        textcoords="offset points",
        color=ORANGE,
    )
    b.annotate("ask move above its\npermanent level", (0.3, 0.1), color=BLUE)
    b.set_xticks(x, ["0", "1/32", "1/16", "1/8", "1/4", "1/2", "1", "2", "4", "8"])
    b.set_xlabel("minutes since a market buy", color=INK)
    b.set_ylabel("fraction of the lag-zero impact left", color=INK)
    b.set_title("two resilience rates (dashed: exponential fits)", color=INK, loc="left")
    b.legend(
        frameon=False, labelcolor=INK, loc="upper right", title="shares bought", title_fontsize=9
    )
    for name in PLANS:
        m = [table[T][name]["sim"] for T in HORIZONS]
        c.plot(HORIZONS, m, color=COLORS[name], marker="o", markersize=4, label=name)
    c.set_title(f"buying {X} shares in {N + 1} trades", color=INK, loc="left")
    c.set_ylabel("simulated shortfall per share (ticks)", color=INK)
    c.legend(frameon=False, labelcolor=INK, loc="lower left")
    d.axhline(0, color=INK, linewidth=0.8)
    for name in PLANS[2:]:
        m = np.array([table[T][name]["saving"][0] for T in HORIZONS])
        se = np.array([table[T][name]["saving"][1] for T in HORIZONS])
        pred = [table[T][name]["model"] for T in HORIZONS]
        d.errorbar(HORIZONS, m, 2 * se, color=COLORS[name], marker="o", markersize=4, label=name)
        d.plot(HORIZONS, pred, color=COLORS[name], linestyle="--", linewidth=1.1)
    d.set_title("saving over even slices: own model (dashed) vs book", color=INK, loc="left")
    d.set_ylabel("ticks a share, simulated with 2 se", color=INK)
    d.legend(frameon=False, labelcolor=INK, loc="upper left")
    for ax in (c, d):
        ax.set_xscale("log")
        ax.set_xticks(HORIZONS, [f"{t:g}" for t in HORIZONS])
        ax.set_xlabel("horizon T (minutes)", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    t0 = time.perf_counter()
    with ProcessPoolExecutor() as pool:
        snaps = np.array(list(pool.map(snapshot, range(400), chunksize=10)), dtype=float)
        depth = snaps.reshape(-1, LEVELS).mean(0)
        print("mean ask depth by tick above the touch, 400 sessions x 20 snapshots")
        print(" ".join(f"{v:.2f}" for v in depth))
        pooled = []
        for q in (1, 2, 3, 4):
            r = np.array(list(pool.map(respond, [(q, s) for s in range(2000)], chunksize=25)))
            pooled.append(r[~np.isnan(r).any(axis=(1, 2))] / q)
        resp = np.concatenate(pooled)[:, :, 1]
        ask = resp.mean(0)
        ow = fit_response(LAGS, ask, resp.std(0, ddof=1) / np.sqrt(len(resp)))
        print(
            f"\nOW fit to the ask: permanent {ow.lam:.3f}, transient {ow.kappa:.3f},"
            f" rho {ow.rho:.2f} (half-life {np.log(2) / ow.rho:.3f} min)"
        )
        print("lag    " + "".join(f"{lag:>8g}" for lag in LAGS))
        print("ask    " + "".join(f"{v:8.3f}" for v in ask))
        print("fit    " + "".join(f"{v:8.3f}" for v in ow.response(LAGS)))
        print("\nmissing ask volume per share bought, 2000 paired runs per size")
        gone = {}
        for q in SWEEPS:
            g = np.array(list(pool.map(refill, [(q, s) for s in range(2000)], chunksize=25)))
            gone[q] = g.mean(0) / q
            print(f"{q:<6d} " + "".join(f"{v:8.3f}" for v in gone[q]))
        fit = (LAGS > 0) & (LAGS <= 2)
        left = np.mean([m / m[0] for m in gone.values()], axis=0)
        rho_e = -np.sum(LAGS[fit] * np.log(left[fit])) / np.sum(LAGS[fit] ** 2)
        print(f"refill rate {rho_e:.3f} a minute (half-life {np.log(2) / rho_e:.2f} min)")
        shape = Profile(tuple(depth))
        table = {}
        print(f"\n{X} shares, {N + 1} trades, {RUNS} paired runs; ticks a share")
        print(f"{'T':6s} {'plan':11s} {'lots':24s} {'model':>6s} {'sim':>6s}  vs even      short")
        for T in HORIZONS:
            late = (*np.linspace(0.0, T, N + 1), *(T + RETRY))
            schedules = {name: done(xi) for name, xi in plans(T, ow, rho_e, shape).items()}
            runs = {}
            for name, cum in schedules.items():
                jobs = [(late, (*cum, *[X] * len(RETRY)), s) for s in range(RUNS)]
                runs[name] = np.array(list(pool.map(race, jobs, chunksize=50)))
            table[T] = {}
            for name, cum in schedules.items():
                paid, got = runs[name].T
                lots = np.diff(cum, prepend=0)
                pred = (
                    predicted(name, lots, T, ow, rho_e, shape)
                    - predicted(name, np.diff(schedules["even"], prepend=0), T, ow, rho_e, shape)
                ) / X
                table[T][name] = {
                    "sim": paid.mean() / X,
                    "saving": mean_se((paid - runs["even"][:, 0]) / X),
                    "model": pred,
                }
                m, se = table[T][name]["saving"]
                print(
                    f"{T:<6g} {name:11s} {' '.join(map(str, lots)):24s} {pred:+6.3f}"
                    f" {paid.mean() / X:6.3f}  {m:+.3f} ({se:.3f})  {np.mean(got < X):.4f}"
                )
    chart(depth, ow, ask, gone, rho_e, table, ROOT / "docs" / "short_execution.png")
    print(f"\n{time.perf_counter() - t0:.0f} s")
