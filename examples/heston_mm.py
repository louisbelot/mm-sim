from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmsim import heston as h
from mmsim.multi import Logistic, Market

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, GREEN, RED = "#2a78d6", "#eb6834", "#1baf7a", "#d2473f"
ROOT = Path(__file__).resolve().parents[1]

Q = h.Heston(3.0, 0.0225, 0.2, -0.5)
P = h.Heston(2.0, 0.04, 0.2, -0.5)
S0, NU0, NU = 10.0, 0.0225, (0.0144, 0.0324)
K = np.repeat([8.0, 9.0, 10.0, 11.0, 12.0], 4)
T = np.tile([1.0, 1.5, 2.0, 3.0], 5)
VBAR, HORIZON = 1e7, 0.0012
GAMMAS = (1e-3, 1e-4, 1e-5, 1e-6)
DAY = 1 / 252
PATHS = 1000
STEPS = 100
PAPER = {
    (8, 1.0): (2.06, 0.41, 0.1622), (8, 1.5): (2.11, 0.46, 0.1583),
    (8, 2.0): (2.16, 0.48, 0.1563), (8, 3.0): (2.27, 0.46, 0.1544),
    (9, 1.0): (1.21, 0.91, 0.1538), (9, 1.5): (1.31, 0.84, 0.1518),
    (9, 2.0): (1.41, 0.76, 0.1515), (9, 3.0): (1.57, 0.65, 0.1508),
    (10, 1.0): (0.58, 1.25, 0.1460), (10, 1.5): (0.71, 1.05, 0.1459),
    (10, 2.0): (0.83, 0.92, 0.1468), (10, 3.0): (1.02, 0.76, 0.1476),
    (11, 1.0): (0.21, 1.05, 0.1392), (11, 1.5): (0.33, 0.96, 0.1409),
    (11, 2.0): (0.44, 0.88, 0.1427), (11, 3.0): (0.62, 0.75, 0.1448),
    (12, 1.0): (0.06, 0.54, 0.1344), (12, 1.5): (0.13, 0.65, 0.1367),
    (12, 2.0): (0.21, 0.67, 0.1395), (12, 3.0): (0.36, 0.65, 0.1423),
}  # fmt: skip

pr = h.Pricer.of(Q, K, nu_lo=NU[0])
g = pr.greeks(S0, NU0, T)
V = pr.vega(S0, NU0, T)
lam = 252 * 30 / (1 + 0.7 * np.abs(S0 - K))
z = 5e5 / g.price
flow = tuple(Logistic(a, 0.7, 150 / v) for a, v in zip(lam, V, strict=True))


def market(gamma: float, mv: bool = False) -> Market:
    eta = P.xi / 2 * np.sqrt(1 - P.rho**2 * mv)
    return Market(eta**2 * np.outer(V, V), flow, gamma, z, np.zeros(len(K)))


def solve(gamma: float, drift: bool, mv: bool = False, points: int = 201) -> h.Reduced:
    m = market(gamma, mv)
    if drift:
        return h.Reduced(m, V, P, Q, *NU, VBAR, HORIZON, rows=31, points=points, mv=mv)
    return h.Reduced(m, V, P, P, *NU, VBAR, HORIZON, rows=1, points=points, mv=mv)


def run(job):
    gamma, kind, mv = job
    m = market(gamma, mv)
    if kind == "closed":

        def pol(tau, nu, n, vega):
            return m.quotes(n * z * vega / V, min(tau, HORIZON))

    else:
        pol = solve(gamma, kind == "drift", mv).policy
    s = h.simulate(pr, T, m, pol, P=P, S0=S0, nu0=NU0, T=DAY, dt=DAY / STEPS, paths=PATHS,
                   vbar=VBAR, mv=mv, seed=11)  # fmt: skip
    return job, s


if __name__ == "__main__":
    print("Heston chain, S0 10, nu0 0.0225 (ours, then the paper's Figure 4 legend)")
    for i in range(len(K)):
        p, v, s = PAPER[(int(K[i]), T[i])]
        vol = h.iv(g.price[i], S0, K[i], T[i])
        print(
            f"  K {K[i]:4.0f} T {T[i]:.1f}  px {g.price[i]:.4f} ({p:.2f})  "
            f"vega {V[i]:.4f} ({v:.2f})  iv {vol:.4f} ({s:.4f})"
        )

    base = solve(1e-3, True, points=401)
    flat = solve(1e-3, False, points=401)
    print(f"\nvalue at V = 0: no drift {flat.value(NU0):,.0f}")
    for nu in (NU[0], NU0, NU[1]):
        print(f"  nu {nu:.4f}: {base.value(nu):,.2f}  edge {base.edge(nu):.4f} per unit vega-year")
    b0, a0 = flat.quotes(HORIZON, NU0, 0.0)
    for nu in NU:
        b, a = base.quotes(HORIZON, nu, 0.0)
        db, da = np.abs(b / b0 - 1).max(), np.abs(a / a0 - 1).max()
        print(f"  nu {nu}: bid moves {db:.2e}, ask {da:.2e}")
    bh, _ = flat.quotes(HORIZON / 2, NU0, 0.0)
    print(f"  quotes at tau = T/2 vs T: {np.abs(bh / b0 - 1).max():.2e}")

    x = h.own(base, base.quotes, nu0=NU0, dt=HORIZON / 200, paths=8000, seed=1)
    print(f"  own-model MC {x.mean():,.0f} (se {x.std() / np.sqrt(len(x)):,.0f})")

    for gamma in GAMMAS:
        print(f"  gamma {gamma:.0e}: target vega edge/(gamma eta^2) = "
              f"{base.edge(NU0) / (gamma * (P.xi / 2) ** 2):,.0f}")  # fmt: skip

    jobs = [(gm, kind, mv) for gm in GAMMAS for kind in ("drift", "flat") for mv in (False, True)]
    jobs += [(gm, "closed", False) for gm in GAMMAS]
    with ProcessPoolExecutor(8) as ex:
        out = dict(ex.map(run, jobs))

    print(f"\none day under P, {PATHS} paths, {STEPS} steps")
    for job, s in out.items():
        obj, se = s.objective.mean(), s.objective.std() / np.sqrt(PATHS)
        t = s.terms
        print(
            f"  gamma {job[0]:.0e} {job[1]:6s} {'mv' if job[2] else 'delta':5s}  "
            f"objective {obj:10,.0f} (se {se:6,.0f})  pnl {s.pnl.mean():10,.0f}  "
            f"sd {s.pnl.std():8,.0f}  spread {t['spread'].mean():10,.0f}  "
            f"vega+delta {(t['vega'] + t['delta']).mean():7,.0f} "
            f"(sd {(t['vega'] + t['delta'] + t['other']).std():7,.0f})  "
            f"end vega {s.vega.mean():11,.0f}  penalty {s.penalty.mean():9,.0f}"
        )

    def paired(a, b):
        d = out[a].objective - out[b].objective
        return d.mean(), d.std() / np.sqrt(PATHS)

    def hedgeable(job):
        t = out[job].terms
        return (t["vega"] + t["delta"] + t["other"]).var()

    print("\npaired differences in objective (mean, se)")
    gain = []
    for gm in GAMMAS:
        drift = paired((gm, "drift", False), (gm, "flat", False))
        mv = paired((gm, "drift", True), (gm, "drift", False))
        cf = paired((gm, "drift", False), (gm, "closed", False))
        ratio = hedgeable((gm, "drift", True)) / hedgeable((gm, "drift", False))
        gain.append((*mv, ratio))
        print(
            f"  gamma {gm:.0e}: drift-aware {drift[0]:7,.0f} ({drift[1]:5,.0f})  "
            f"mv hedge {mv[0]:7,.0f} ({mv[1]:5,.0f}), variance ratio {ratio:.3f}  "
            f"HJB over closed form {cf[0]:7,.0f} ({cf[1]:5,.0f})"
        )

    bars = np.linspace(-VBAR, VBAR, 201)
    bid, _ = base.quotes(HORIZON, np.full(len(bars), NU0), bars)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    for i, c in zip((8, 9, 10, 11), (BLUE, ORANGE, GREEN, RED), strict=True):
        ok = np.isfinite(bid[:, i])
        label = f"K 10, T {T[i]:.1f}"
        ax.plot(bars[ok] / 1e6, bid[ok, i] / g.price[i], color=c, lw=1.8, label=label)
    ax.set_xlabel("portfolio vega (millions)")
    ax.set_ylabel("mid-to-bid / option price")
    ax.set_title("Bid quotes, gamma 1e-3, nu 0.0225", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9)
    xs = np.arange(len(GAMMAS))
    mean, se, ratio = (np.array(a) for a in zip(*gain, strict=True))
    bx.bar(xs, mean / 1e3, 0.55, color=BLUE, yerr=2 * se / 1e3, ecolor=INK, capsize=4)
    for x, m, e, r in zip(xs, mean, se, ratio, strict=True):
        bx.annotate(f"var x{r:.2f}", (x, (m + 2 * e) / 1e3), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=9, color=MUTED)  # fmt: skip
    bx.set_ylim(top=bx.get_ylim()[1] * 1.12)
    bx.set_xticks(xs, [f"{gm:.0e}" for gm in GAMMAS])
    bx.set_xlabel("risk aversion gamma")
    bx.set_ylabel("objective gain per day (thousands)")
    bx.set_title("Minimum-variance hedge over delta hedge", color=INK, fontsize=11, loc="left")
    for a in (ax, bx):
        a.set_facecolor(SURFACE)
        a.grid(color=GRID, lw=0.8)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.tick_params(colors=MUTED)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "heston_mm.png", dpi=150)
