import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from rl_market_maker import BLUE, GREEN, INK, MUTED, ORANGE, ROOT, SURFACE, style

from mmsim.crash import Maker, Rule, Value, run, session
from mmsim.flow import CSTParams, ZeroIntelligence

SKY = CSTParams.sky_perfect()
QTY, RATE, MINUTES, START = 300, 0.45, 90.0, 5.0
STOPS = tuple((d, 2) for d in range(3, 60, 2))
CRASH = 30.0
COOL = 1.0
RATES = (0.15, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6)
LIMITS = (4, 6, 8, 10, 14, 20, 30, 50, None)
BANDS = (5, 10, 20, 40)
SEEDS = range(300)
PURPLE = "#7b5cc4"

RULES = {
    "none": Rule(),
    "collar": Rule(band=10),
    "LULD": Rule(band=10, limit=0.25),
    "trip and halt": Rule(trip=10),
}
COLORS = dict(zip(RULES, (INK, ORANGE, BLUE, GREEN), strict=True))


def flow():
    return ZeroIntelligence(SKY)


def makers(limit=10, dump=True, n=5):
    return lambda: [
        Maker(1, 2, 0.1, limit, limit if dump else None, 4, cool=COOL) for _ in range(n)
    ]


def value(seed):
    return [Value(3.0, 0.5, 10, seed * 10 + i) for i in range(4)]


def play(args):
    rule, seed, kw = args
    kw = dict(kw)
    mm = makers(kw.pop("limit", 10), kw.pop("dump", True), kw.pop("n", 5))
    base = dict(flow=flow, makers=mm, value=value, qty=QTY, rate=RATE, stops=STOPS)
    base.update(kw)
    return session(seed, rule, minutes=MINUTES, start=START, **base)


def trace(rule, seed):
    r = run(
        seed,
        rule,
        flow=flow,
        makers=makers(),
        value=value,
        qty=QTY,
        rate=RATE,
        stops=STOPS,
        minutes=MINUTES,
        start=START,
    )
    t = np.asarray(r.sim.tape.t) - r.t0
    m = np.asarray(r.sim.tape.mid) - r.arrival
    keep = t >= 0
    return t[keep], m[keep], [(a - r.t0, b - r.t0) for a, b in r.guard.pauses if b > r.t0]


def grid(pool, jobs):
    return list(pool.map(play, jobs, chunksize=8))


def stats(rows):
    drop = np.array([r.drop for r in rows])
    return dict(
        median=np.median(drop),
        p90=np.percentile(drop, 90),
        crash=(drop >= CRASH).mean(),
        far=np.mean([r.far for r in rows]),
        cost=np.mean([r.cost for r in rows if r.sold]) if any(r.sold for r in rows) else np.nan,
        sold=np.mean([r.sold for r in rows]),
        makers=np.mean([r.makers for r in rows]),
        dumped=np.mean([r.dumped for r in rows]),
        pauses=np.mean([r.pauses for r in rows]),
        halted=np.mean([r.halted for r in rows]),
        recover=np.nanmedian([r.recover for r in rows]),
    )


def se(p, n):
    return np.sqrt(p * (1 - p) / n)


def chart(path, by_rate, parts, limits, traces):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9), facecolor=SURFACE)
    ax = axes[0, 0]
    t, m, pauses = traces["LULD"]
    for a, b in pauses:
        ax.axvspan(max(a, 0), min(b, MINUTES), color=BLUE, alpha=0.12, lw=0)
    for name in ("none", "LULD"):
        t, m, _ = traces[name]
        ax.step(t, m, where="post", color=COLORS[name], lw=1.3, label=name)
    ax.axvline(START, color=MUTED, ls=":", lw=1)
    ax.set(xlabel="minutes", ylabel="mid minus value (ticks)", xlim=(0, MINUTES))
    ax.set_title("one session, same seed (shaded: LULD pauses)", color=INK, loc="left")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[0, 1]
    n = len(SEEDS)
    for name, rows in by_rate.items():
        p = np.array([stats(r)["crash"] for r in rows])
        ax.errorbar(
            RATES,
            p,
            yerr=2 * se(p, n),
            color=COLORS[name],
            marker="o",
            ms=4,
            capsize=2,
            lw=1.3,
            label=name,
        )
    ax.set(xlabel="seller's participation rate", ylabel=f"P(drop >= {CRASH:g} ticks)")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("crash probability against the seller's speed", color=INK, loc="left")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1, 0]
    labels = list(parts)
    data = [[r.drop for r in parts[k]] for k in labels]
    bp = ax.boxplot(data, orientation="horizontal", widths=0.6, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set(facecolor="#e9eef8", edgecolor=MUTED)
    for line in bp["medians"]:
        line.set(color=INK, lw=1.5)
    ax.set_yticks(range(1, len(labels) + 1), labels)
    ax.invert_yaxis()
    ax.set(xlabel="worst print below value (ticks), no breaker")
    ax.set_title(f"what the crash needs (rate {RATE:g})", color=INK, loc="left")

    ax = axes[1, 1]
    xs = np.arange(len(LIMITS))
    med = [np.median([r.drop for r in rows]) for rows in limits]
    q1 = [np.percentile([r.drop for r in rows], 25) for rows in limits]
    q3 = [np.percentile([r.drop for r in rows], 75) for rows in limits]
    ax.fill_between(xs, q1, q3, color=PURPLE, alpha=0.15, lw=0)
    ax.plot(xs, med, color=PURPLE, marker="o", ms=4, lw=1.5, label="median, quartiles")
    ax.set_xticks(xs, [str(x) if x is not None else "none" for x in LIMITS])
    ax.set(xlabel="maker inventory limit (shares)", ylabel="worst print below value (ticks)")
    ax.set_title("crash depth against the makers' limit", color=INK, loc="left")
    ax.legend(frameon=False, loc="upper right")

    for ax in axes.flat:
        style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)


if __name__ == "__main__":
    t0 = time.perf_counter()
    with ProcessPoolExecutor() as pool:
        by_rate = {
            name: [grid(pool, [(rule, s, {"rate": x}) for s in SEEDS]) for x in RATES]
            for name, rule in RULES.items()
        }
        print(f"{len(SEEDS)} sessions a cell, {QTY} shares to sell, {MINUTES:g} minutes")
        print("crash probability by participation rate")
        print("rule           " + " ".join(f"{x:>6g}" for x in RATES))
        for name, rows in by_rate.items():
            print(f"{name:14s} " + " ".join(f"{stats(r)['crash']:6.2f}" for r in rows))

        base = {name: rows[RATES.index(RATE)] for name, rows in by_rate.items()}
        calm = {
            name: grid(pool, [(rule, s, {"qty": 0}) for s in SEEDS]) for name, rule in RULES.items()
        }
        print(f"\nat rate {RATE:g}")
        print(
            "rule           median   p90  crash  far<=-20  cost/sh  sold  makers  dumped"
            "  pauses  halted  recover  calm pauses"
        )
        for name, rows in base.items():
            s = stats(rows)
            c = stats(calm[name])
            print(
                f"{name:14s} {s['median']:6.1f} {s['p90']:5.1f} {s['crash']:6.2f}"
                f" {s['far']:9.1f} {s['cost']:8.2f} {s['sold']:5.0f} {s['makers']:7.0f}"
                f" {s['dumped']:7.1f} {s['pauses']:7.2f} {s['halted']:7.2f}"
                f" {s['recover']:8.1f} {c['pauses']:12.3f}"
            )

        bands = {}
        for b in BANDS:
            rule = Rule(band=b, limit=0.25)
            bands[b] = (
                stats(grid(pool, [(rule, s, {}) for s in SEEDS])),
                stats(grid(pool, [(rule, s, {"qty": 0}) for s in SEEDS])),
            )
        print("\nLULD band width")
        print("band  median  crash  pauses  calm pauses  calm halted")
        for b, (s, c) in bands.items():
            print(
                f"{b:4d} {s['median']:7.1f} {s['crash']:6.2f} {s['pauses']:7.2f}"
                f" {c['pauses']:12.3f} {c['halted']:12.2f}"
            )

        none = RULES["none"]
        parts = {
            "everything": base["none"],
            "no stops": grid(pool, [(none, s, {"stops": ()}) for s in SEEDS]),
            "makers stop at the limit, no dumping": grid(
                pool, [(none, s, {"dump": False}) for s in SEEDS]
            ),
            "no stops, no dumping": grid(
                pool, [(none, s, {"stops": (), "dump": False}) for s in SEEDS]
            ),
            "no makers": grid(pool, [(none, s, {"n": 0}) for s in SEEDS]),
        }
        print(f"\ningredients, no breaker, rate {RATE:g}")
        for k, rows in parts.items():
            s = stats(rows)
            print(f"{k:38s} median {s['median']:6.1f}  crash {s['crash']:.2f}  far {s['far']:6.1f}")

        limits = [grid(pool, [(none, s, {"limit": L or 10**6}) for s in SEEDS]) for L in LIMITS]
        print("\nmaker limit")
        for L, rows in zip(LIMITS, limits, strict=True):
            s = stats(rows)
            print(
                f"{L!s:>5} median {s['median']:6.1f}  crash {s['crash']:.2f}"
                f"  dumped {s['dumped']:6.1f}  makers {s['makers']:8.0f}"
            )

    seed = next(s for s, r in zip(SEEDS, base["none"], strict=True) if r.drop >= 80)
    traces = {name: trace(rule, seed) for name, rule in RULES.items()}
    chart(ROOT / "docs" / "flash_crash.png", by_rate, parts, limits, traces)
    print(f"\nseed {seed} traced; {time.perf_counter() - t0:.0f} s")
