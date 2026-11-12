from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import NamedTuple

import numpy as np
from scipy.optimize import minimize

from . import robust as r

GAME = ("zeta", "nu", "eta")
MARKET = ("A", "k")
BOUNDS = {
    "zeta": (0.0, 0.1),
    "nu": (0.0, 0.2),
    "eta": (0.0, 2.0),
    "A": (50.0, 195.0),
    "k": (0.5, 3.0),
}


class Fit(NamedTuple):
    params: dict[str, float]
    loss: float
    solution: r.Solution


def model(
    game: r.Game, names: Sequence[str], x: Sequence[float], market: r.Market = r.FIXED
) -> tuple[r.Game, r.Market]:
    unknown = set(names) - set(GAME) - set(MARKET)
    if unknown:
        raise ValueError(f"cannot fit {sorted(unknown)}")
    vals = dict(zip(names, (float(v) for v in x), strict=True))
    g = replace(game, **{n: v for n, v in vals.items() if n in GAME})
    m = market._replace(**{n: v for n, v in vals.items() if n in MARKET})
    return g, m


def solve(game: r.Game, names, x, market: r.Market = r.FIXED) -> r.Solution:
    return r.best_response(*model(game, names, x, market))


def quote_error(sol: r.Solution, bid: np.ndarray, ask: np.ndarray, w: np.ndarray) -> float:
    total = 0.0
    for mine, theirs in ((sol.bid, bid), (sol.ask, ask)):
        ok = np.isfinite(mine) & np.isfinite(theirs) & (w > 0)
        total += float((w[ok] * (mine[ok] - theirs[ok]) ** 2).sum())
    return total / float(w.sum())


def start(names: Sequence[str], market: r.Market) -> list[float]:
    return [getattr(market, n) if n in MARKET else BOUNDS[n][1] / 10 for n in names]


def _fit(loss, x0, bounds, fatol):
    opts = {"xatol": 1e-7, "fatol": fatol, "maxiter": 4000}
    return minimize(loss, np.asarray(x0, float), method="Nelder-Mead", bounds=bounds, options=opts)


def match(
    game: r.Game,
    bid: np.ndarray,
    ask: np.ndarray,
    visits: np.ndarray,
    names: Sequence[str] = ("zeta", "nu"),
    x0: Sequence[float] | None = None,
    bounds: dict[str, tuple[float, float]] | None = None,
    market: r.Market = r.FIXED,
) -> Fit:
    def loss(x):
        return quote_error(solve(game, names, x, market), bid, ask, visits)

    box = [(bounds or {}).get(n, BOUNDS[n]) for n in names]
    res = _fit(loss, start(names, market) if x0 is None else x0, box, 1e-12)
    return Fit(
        dict(zip(names, res.x, strict=True)), float(res.fun), solve(game, names, res.x, market)
    )


def qvalues(
    game: r.Game, sol: r.Solution, market: r.Market, n: np.ndarray, i: np.ndarray, deltas
) -> np.ndarray:
    db, da = (x.ravel() for x in np.meshgrid(deltas, deltas, indexing="ij"))
    out = np.empty((len(n), len(db)))
    for t in np.unique(n):
        rows = np.flatnonzero(n == t)
        st = r.stage(game, sol.V[t + 1]).at(i[rows])
        mkt = r.Market(*(r.at(x, t) for x in market))
        out[rows] = r.payoff(st, db[:, None], da[:, None], *mkt).T
    return out


def loglik(q: np.ndarray, a: np.ndarray, beta: float) -> float:
    z = beta * q
    z -= z.max(axis=1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=1))
    return float((z[np.arange(len(a)), a] - lse).sum())


def boltzmann(
    game: r.Game,
    n: np.ndarray,
    i: np.ndarray,
    a: np.ndarray,
    deltas: np.ndarray,
    names: Sequence[str] = ("zeta",),
    x0: Sequence[float] | None = None,
    beta0: float = 10.0,
    market: r.Market = r.FIXED,
) -> Fit:
    k = len(names)

    def loss(x):
        g, m = model(game, names, x[:k], market)
        sol = r.best_response(g, m)
        return -loglik(qvalues(g, sol, m, n, i, deltas), a, np.exp(x[k])) / len(a)

    box = [BOUNDS[nm] for nm in names] + [(-5.0, 10.0)]
    x0 = start(names, market) if x0 is None else list(x0)
    res = _fit(loss, [*x0, np.log(beta0)], box, 1e-10)
    params = dict(zip(names, res.x[:k], strict=True))
    params["beta"] = float(np.exp(res.x[k]))
    return Fit(params, float(res.fun), solve(game, names, res.x[:k], market))
