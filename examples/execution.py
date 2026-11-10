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
from mmsim.bots import GLFT, RegimeLean, SymmetricQuoter
from mmsim.detect import one_sided
from mmsim.execution import (
    AlmgrenChriss,
    Executor,
    fit_response,
    impulse,
    targets,
    trial,
)
from mmsim.flow import CSTParams, ZeroIntelligence
from mmsim.glft import Model
from mmsim.sim import Agent, Simulator

SKY = CSTParams.sky_perfect()
LAGS = np.array([0, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32])
SIZES = (1, 2, 3, 4)
X, T, N = 24, 24.0, 12
URGENCY = (0.0, 0.0005, 0.005, 0.05, 5.0)
START, LENGTH, MINUTES = 20.0, 20.0, 60.0
PARENTS = (0, 5, 10, 20, 30)
SHIFTS = (0.5, 1.0, 2.0, 3.0)
TUNE = range(10_000_000, 10_000_400)
TEST = range(1000)
MODEL = Model(SIGMA, A, K, 0.08, Q)


def sky():
    return ZeroIntelligence(SKY)


def respond(args):
    qty, seed = args
    return impulse(sky, qty, LAGS, seed)


def quiet(seed):
    sim = Simulator(sky(), seed=seed, record_every=None)
    sim.seed(10_000, 10_001)
    sim.warm_up(30.0)
    spread = sim.book.best_ask() - sim.book.best_bid()
    sim.run(T)
    return spread, float(sim.tape.at(sim.t) - sim.tape.at(30.0))


def race(args):
    times, done, seed = args
    bot = Executor(Side.BUY, times, done)
    return trial(seed, {}, bot, flow=sky, minutes=T + 1.0)[1]


def filt():
    return one_sided(SKY.mu, 1.0, 1 / MINUTES, 1 / LENGTH)


def maker(name, shift):
    match name:
        case "none":
            return {}
        case "join the touch":
            return {name: lambda: SymmetricQuoter(0.5, max_position=Q)}
        case "GLFT":
            return {name: lambda: GLFT(MODEL, keep=1)}
        case "GLFT + filter":
            return {name: lambda: GLFT(MODEL, keep=1, signal=RegimeLean(filt(), shift))}


def parent(qty, seed):
    if qty == 0:
        return None
    side = Side.BUY if seed % 2 == 0 else Side.SELL
    times = START + LENGTH * (np.arange(qty) + 0.5) / qty
    return Executor(side, times, np.arange(1, qty + 1))


def meet(args):
    name, shift, qty, seed = args
    res, cost = trial(seed, maker(name, shift), parent(qty, seed), flow=sky, minutes=MINUTES)
    if not res:
        return (0.0,) * 6 + (cost,)
    r = res[name]
    sign = 1 if seed % 2 == 0 else -1
    score = r.pnl - PHI * r.exposure - TERM * r.position**2
    return score, r.pnl, r.spread, r.adverse, r.inventory, sign * r.position, cost


class Watch(Agent):
    def __init__(self):
        self.filter = filt()
        self.path = []

    def on_start(self, ctx):
        self.filter.t = ctx.t
        ctx.set_timer(0.0)

    def on_trade(self, ctx, p):
        self.filter.observe(p.t, p.side, p.qty)

    def on_timer(self, ctx):
        self.path.append(self.filter.at(ctx.t)[1])
        ctx.set_timer(0.25)


def watch(args):
    qty, seed = args
    w = Watch()
    bot = Executor(Side.BUY, START + LENGTH * (np.arange(qty) + 0.5) / qty, np.arange(1, qty + 1))
    sim = Simulator(sky(), [w, bot], seed=seed, record_every=None)
    sim.seed(10_000, 10_001)
    sim.warm_up(30.0)
    sim.run(MINUTES)
    return w.path


def mean_se(x, axis=0):
    x = np.asarray(x, dtype=float)
    return x.mean(axis), x.std(axis, ddof=1) / np.sqrt(x.shape[axis])


def chart(resp, fit, frontier, table, paths, path):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4), facecolor=SURFACE)
    for ax in axes.flat:
        style(ax)
    (a, b), (c, d) = axes

    m, se = resp
    x = np.arange(len(LAGS))
    for j, (name, color) in enumerate((("bid", BLUE), ("ask", ORANGE))):
        a.errorbar(x, m[:, j], 2 * se[:, j], color=color, marker="o", markersize=4, label=name)
    a.plot(x, fit.response(LAGS), color=INK, linestyle="--", linewidth=1.1, label="OW fit to ask")
    a.set_xticks(x, [f"{lag:g}" for lag in LAGS])
    a.set_ylim(0, None)
    a.set_xlabel("minutes since a market buy", color=INK)
    a.set_ylabel("quote move per share bought (ticks)", color=INK)
    a.set_title("impulse response of the zero-intelligence book", color=INK, loc="left")
    a.legend(frameon=False, labelcolor=INK, loc="lower right")

    labels, pred, real = frontier
    ps = np.array([p[1] for p in pred]) ** 0.5
    b.plot(ps, [p[0] / X for p in pred], color=MUTED, marker="s", markerfacecolor="none")
    b.plot([], [], color=MUTED, marker="s", markerfacecolor="none", label="Almgren-Chriss model")
    for lab, p, (mean, se_mean, sd) in zip(labels, pred, real, strict=True):
        b.errorbar(sd, mean / X, 2 * se_mean / X, color=ORANGE, marker="o", markersize=5)
        off = {"even": (6, -14), "0.0005": (-48, 8)}.get(lab, (6, 6))
        b.annotate(lab, (sd, mean / X), xytext=off, textcoords="offset points", color=INK)
        b.plot([p[1] ** 0.5, sd], [p[0] / X, mean / X], color=MUTED, linewidth=0.6, linestyle=":")
    b.plot([], [], color=ORANGE, marker="o", linestyle="none", label="simulated, mean and 2 se")
    b.set_xlim(-3, 86)
    b.set_xlabel("sd of shortfall (ticks)", color=INK)
    b.set_ylabel("mean shortfall per share (ticks)", color=INK)
    b.set_title(f"buying {X} shares in {T:g} minutes, by risk aversion", color=INK, loc="left")
    b.legend(frameon=False, labelcolor=INK, loc="upper right")

    for (name, rows), color in zip(table.items(), (MUTED, BLUE, GREEN), strict=True):
        mean = [r[0][0] for r in rows]
        err = [2 * r[1][0] for r in rows]
        c.errorbar(PARENTS, mean, err, color=color, marker="o", markersize=4, label=name)
    c.axhline(0, color=MUTED, linewidth=0.8)
    c.set_xticks(PARENTS)
    c.set_xlabel("shares the executor buys or sells in 20 of the 60 minutes", color=INK)
    c.set_ylabel("market maker score per session (ticks)", color=INK)
    c.set_title("what the executor costs the market maker", color=INK, loc="left")
    c.legend(frameon=False, labelcolor=INK, loc="lower left")

    t = np.arange(len(next(iter(paths.values())))) * 0.25
    d.axvspan(START, START + LENGTH, color=ORANGE, alpha=0.08, linewidth=0)
    for (qty, p), color in zip(paths.items(), (MUTED, BLUE, GREEN, ORANGE), strict=True):
        d.plot(t, p, color=color, linewidth=1.4, label=f"{qty} shares")
    d.set_ylim(0, 1)
    d.set_xlabel("minute of the session (buyer active in the shaded window)", color=INK)
    d.set_ylabel("mean posterior probability of a buyer", color=INK)
    d.set_title("the flow filter finding the executor", color=INK, loc="left")
    d.legend(frameon=False, labelcolor=INK, loc="upper left")

    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    t0 = time.perf_counter()
    with ProcessPoolExecutor() as pool:
        print("quote move per share after a market buy (ticks), 2000 paired runs per size")
        print("size   " + "".join(f"{lag:>7g}" for lag in LAGS))
        pooled = []
        for q in SIZES:
            d = np.array(list(pool.map(respond, [(q, s) for s in range(2000)], chunksize=25)))
            d = d[~np.isnan(d).any(axis=(1, 2))] / q
            pooled.append(d)
            for j, name in enumerate(("bid", "ask")):
                print(f"{q} {name}  " + "".join(f"{v:7.3f}" for v in d[:, :, j].mean(0)))
        resp = mean_se(np.concatenate(pooled))
        for j, name in enumerate(("bid", "ask")):
            print(f"all {name} " + "".join(f"{v:7.3f}" for v in resp[0][:, j]))
        fit = fit_response(LAGS, resp[0][:, 1], resp[1][:, 1])
        print(
            f"OW fit to the ask: permanent {fit.lam:.3f}, transient {fit.kappa:.3f}, "
            f"rho {fit.rho:.2f} (half-life {np.log(2) / fit.rho:.3f})"
        )

        calm = np.array(list(pool.map(quiet, range(4000), chunksize=50)))
        half = calm[:, 0].mean() / 2
        sigma = calm[:, 1].std() / np.sqrt(T)
        slope = fit.lam + fit.kappa
        ac = AlmgrenChriss(sigma, 0.5 * slope * T / N, fit.lam, half - 0.5 * slope)
        print(f"\nhalf spread {half:.3f}, sigma {sigma:.3f} per sqrt minute")
        print(f"{X} shares in {T:g} minutes, {N} slices, 4000 runs each")
        print("schedule         first  model E/share  sd     sim E/share (se)   sd")
        grid = np.linspace(0.0, T, N + 1)
        plans = {f"AC {lam:g}": ac.holdings(X, T, N, lam) for lam in URGENCY}
        labels, pred, real = [], [], []
        for name, x in plans.items():
            done = targets(x)[1:]
            E, V = ac.cost(X - np.concatenate([[0], done]), T / N)
            jobs = [(tuple(grid[:-1]), tuple(done), s) for s in range(4000)]
            c = np.array(list(pool.map(race, jobs, chunksize=50)))
            m, se = mean_se(c)
            print(
                f"{name:15s} {done[0]:5d}  {E / X:10.3f}  {V**0.5:6.2f}"
                f"  {m / X:8.3f} ({se / X:.3f})  {c.std():6.2f}"
            )
            labels.append("even" if name == "AC 0" else name.removeprefix("AC "))
            pred.append((E, V))
            real.append((m, se, c.std()))
        fixed = ac.eps * X
        done = np.rint(np.cumsum(fit.trades(X, T, N))).astype(int)
        jobs = [(tuple(grid), tuple(done), s) for s in range(4000)]
        m, se = mean_se(list(pool.map(race, jobs, chunksize=50)))
        even = fit.cost(np.full(N, X / N), grid[:-1]) + fixed
        print(
            f"OW model per share: continuous optimum {(fit.optimum(X, T) + fixed) / X:.3f}, "
            f"{N + 1} OW trades {(fit.cost(fit.trades(X, T, N), grid) + fixed) / X:.3f}, "
            f"{N} even slices {even / X:.3f}; OW trades simulated {m / X:.3f} ({se / X:.3f})"
        )

        tune = {
            s: mean_se(
                list(pool.map(meet, [("GLFT + filter", s, 20, i) for i in TUNE], chunksize=20))
            )
            for s in SHIFTS
        }
        shift = max(tune, key=lambda s: tune[s][0][0])
        print(
            "\nshift sweep at 20 shares, 400 tuning sessions: "
            + ", ".join(f"{s:g}: {m[0]:.2f} ({e[0]:.2f})" for s, (m, e) in tune.items())
        )
        print(
            f"\n{len(TEST)} held-out sessions, executor active from {START:g} to {START + LENGTH:g}"
        )
        print("maker           shares  score (se)     pnl    spread adverse inventory  pos   exec")
        table = {}
        for name in ("join the touch", "GLFT", "GLFT + filter"):
            table[name] = []
            for qty in PARENTS:
                r = mean_se(
                    list(pool.map(meet, [(name, shift, qty, i) for i in TEST], chunksize=20))
                )
                table[name].append(r)
                m, se = r
                print(
                    f"{name:15s} {qty:5d}  {m[0]:6.2f} ({se[0]:.2f})  {m[1]:6.2f}  {m[2]:6.2f}"
                    f"  {m[3]:6.2f}  {m[4]:7.2f}  {m[5]:5.2f}  {m[6] / max(qty, 1):7.3f}"
                )
        for qty in PARENTS[1:]:
            m, se = mean_se(
                list(pool.map(meet, [("none", 0.0, qty, i) for i in TEST], chunksize=20))
            )
            print(
                f"no maker        {qty:5d}  exec {m[6] / qty:.3f} ({se[6] / qty:.3f}),"
                f" linear impact {half + fit.lam * (qty - 1) / 2:.3f}"
            )

        paths = {}
        for qty in (5, 10, 20, 30):
            p = np.array(list(pool.map(watch, [(qty, i) for i in range(400)], chunksize=20)))
            paths[qty] = p.mean(0)
            t = np.arange(p.shape[1]) * 0.25
            inside = (t >= START) & (t < START + LENGTH)
            hit = p[:, inside] > 0.5
            delay = np.where(hit.any(1), hit.argmax(1) * 0.25, np.nan)
            print(
                f"{qty:2d} shares: posterior over 0.5 in {hit.any(1).mean():.0%} of sessions, "
                f"median delay {np.nanmedian(delay):.1f}; before the window it is over 0.5 "
                f"{(p[:, t < START] > 0.5).mean():.1%} of the time"
            )

    chart(resp, fit, (labels, pred, real), table, paths, ROOT / "docs" / "execution.png")
    print(f"\ndone in {time.perf_counter() - t0:.0f} s")
