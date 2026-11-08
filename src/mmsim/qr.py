from __future__ import annotations

import math
import random
from bisect import bisect_right
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from itertools import accumulate, count

import numpy as np
from scipy.optimize import minimize

from .book import Book, Event, EventType, Level, Order, OrderType, Side
from .hawkes import decayed_at
from .itch import NS_PER_MINUTE, Msg
from .replay import ReplayBook, batches, on_grid

LIMIT, CANCEL, MARKET = range(3)
THIN, USUAL, DEEP = range(3)
STAYED_CANCEL, STAYED_MARKET, LEFT_CANCEL, LEFT_MARKET = range(3, 7)
BASE = 7
REGIMES = 2 * BASE
POSITIONS = 1000
FAR = 50.0
UNIT, STATIONARY, QUEUE = "unit", "stationary", "queue"
KINDS, SHARED = "kinds", "shared"


class Gap:
    __slots__ = ("ask", "wide", "side", "market", "mid", "up")

    def __init__(self) -> None:
        self.ask = math.inf
        self.wide = False
        self.side = Side.BUY
        self.market = False
        self.mid = 0
        self.up = False

    def update(self, bid: int, ask: int, market: bool) -> None:
        mid = bid + ask
        if mid != self.mid:
            self.up = mid > self.mid
            self.mid = mid
        wide = ask - bid > 1
        if wide and not self.wide:
            self.side = Side.SELL if ask > self.ask else Side.BUY
            self.market = market
        elif not wide:
            self.ask = ask
        self.wide = wide

    def regime(self, side: Side) -> int:
        return (LEFT_CANCEL if side == self.side else STAYED_CANCEL) + self.market


@dataclass(frozen=True)
class Clock:
    edges: tuple[float, ...]
    scale: tuple[tuple[float, float, float], ...]

    def slot(self, t: float) -> tuple[int, float]:
        H = len(self.scale)
        h = bisect_right(self.edges, t, 1, H) - 1
        return h, self.edges[h + 1] - t if h < H - 1 else math.inf


@dataclass(frozen=True)
class Reaction:
    own: tuple[float, ...]
    cross: tuple[float, ...]
    rest: float
    cells: dict[int, float]


@dataclass(frozen=True)
class Excite:
    own: tuple[float, ...]
    cross: tuple[float, ...]
    beta: tuple[float, ...]
    react: dict[int, Reaction] | None = None

    @property
    def branching(self) -> float:
        return sum(self.own) + sum(self.cross)


@dataclass(frozen=True)
class Params:
    rates: tuple
    sizes: dict
    aes: tuple[int, ...]
    edges: tuple[int, int]
    queues: tuple[tuple[int, ...], ...]
    conditioned: bool
    back: tuple[tuple[float, ...], ...] | None = None
    memory: bool = False
    clock: Clock | None = None
    excite: Excite | None = None
    regimes: bool = True

    @property
    def K(self) -> int:
        return len(self.aes)

    @property
    def N(self) -> int:
        return len(self.rates[0][0]) - 1


@dataclass(frozen=True)
class Fit:
    T: float
    counts: np.ndarray
    tau: np.ndarray
    aes: np.ndarray
    edges: tuple[int, int]
    events: np.ndarray
    queues: np.ndarray
    slots: np.ndarray
    exposure: np.ndarray
    arrivals: np.ndarray
    t: np.ndarray | None = None
    h: np.ndarray | None = None
    top: np.ndarray | None = None
    hits: np.ndarray | None = None
    quotes: np.ndarray | None = None

    def params(
        self,
        sizes: str = QUEUE,
        regimes: bool = True,
        back: bool = True,
        memory: bool = True,
        min_events: int = 30,
        clock: str | None = None,
    ) -> Params:
        c = None if clock is None else self.clock(clock)
        if c is None:
            tau = np.repeat(self.tau[..., None], 3, axis=-1)
        else:
            tau = np.einsum("irnh,hk->irnk", self.exposure, np.array(c.scale))
        pooled_n, pooled_t = self.counts.sum(axis=1), tau.sum(axis=1)
        out = []
        for i in range(self.aes.size):
            levels = []
            for r in range(REGIMES):
                if not regimes:
                    n, t = pooled_n[i], pooled_t[i]
                elif memory:
                    n, t = self.counts[i, r], tau[i, r]
                else:
                    b = r % BASE
                    n = self.counts[i, b] + self.counts[i, b + BASE]
                    t = tau[i, b] + tau[i, b + BASE]
                levels.append(_filled(n, t, pooled_n[i], pooled_t[i], min_events))
            out.append(tuple(levels))
        return Params(
            tuple(out),
            self._sizes(sizes, regimes, memory, min_events),
            tuple(int(a) for a in self.aes),
            self.edges,
            tuple(tuple(int(v) for v in row) for row in self.queues.T),
            sizes == QUEUE,
            self._back() if back else None,
            memory and regimes,
            c,
            regimes=regimes,
        )

    def clock(self, mode: str = KINDS, tol: float = 1e-10, iters: int = 10_000) -> Clock:
        N = self.counts.reshape(-1, 3)
        T = self.exposure.reshape(-1, self.slots.size - 1)
        L = np.diff(self.slots)
        s = np.ones((L.size, 3))
        for _ in range(iters):
            lam = _ratio(N, T @ s)
            new = _scale(self.arrivals, T.T @ lam, mode)
            new /= L @ new / L.sum()
            done = np.abs(new - s).max() < tol
            s = new
            if done:
                break
        return _clock(self.slots, s)

    def excite(self, p: Params, kernels: int = 1, min_events: int = 30) -> Params:
        p = self._wet(p)
        mu, dt, rows, side, key = self._path(p)
        t = self.t[rows]
        ev = key[rows, side]
        cells, n = np.unique(ev, return_counts=True)
        cells = cells[n >= min_events]

        def group(x):
            return _group(cells, x)

        G = cells.size + 1
        M = np.bincount(group(key).ravel(), (mu * dt[:, None]).ravel(), G)
        g_ev = group(ev)
        free = np.bincount(g_ev, minlength=G) > 0
        mu_ev = mu[rows, side]
        live = key >= 0
        k = kernels
        beta0 = [60.0] if k == 1 else np.geomspace(600.0, 6.0, k)
        x0 = np.concatenate((np.full(free.sum(), 0.5), np.full(2 * k, 0.1 / k), beta0))

        def unpack(x):
            psi = np.ones(G)
            psi[free] = x[: free.sum()]
            return psi, x[free.sum() :]

        def f(x):
            psi, th = unpack(np.exp(x))
            ll, g_psi, g = _excite_score(psi, th, M, g_ev, mu_ev, t, side, self.t, live)
            return -ll, -np.concatenate((g_psi[free], g)) * np.exp(x)

        res = minimize(f, np.log(x0), jac=True, method="L-BFGS-B")
        if not res.success:
            raise RuntimeError(f"excitation fit did not converge: {res.message}")
        psi, th = unpack(np.exp(res.x))
        own, cross, beta = (tuple(float(v) for v in th[j * k : (j + 1) * k]) for j in range(3))
        rates = np.array(p.rates)
        grid = _keys(p, *np.indices(rates.shape[:3]))
        rates[..., MARKET] *= psi[group(grid)]
        return replace(p, rates=_nested(rates), excite=Excite(own, cross, beta))

    def _wet(self, p: Params) -> Params:
        mu, dt, rows, side, key = self._path(p)
        dry = np.unique(key[rows, side][mu[rows, side] == 0])
        if dry.size == 0:
            return p
        i = self.top[..., 0]
        clock = 1.0 if p.clock is None else np.array(p.clock.scale)[self.h, MARKET][:, None]
        live = key >= 0
        exposure = np.bincount(i[live], (dt[:, None] * clock * live)[live], p.K)
        pooled = _ratio(np.bincount(i[rows, side], minlength=p.K), exposure)
        rates = np.array(p.rates)
        lvl = np.indices(rates.shape[:3])
        wet = np.isin(_keys(p, *lvl), dry)
        rates[wet, MARKET] = pooled[lvl[0][wet]]
        return replace(p, rates=_nested(rates))

    def react(
        self, p: Params, kinds: Sequence[int] = (CANCEL, LIMIT), min_events: int = 30
    ) -> Params:
        if p.excite is None:
            raise ValueError("fit the market kernels with excite first")
        bare = replace(p, excite=replace(p.excite, react=None))
        out = {k: self._react(bare, k, min_events) for k in kinds}
        return replace(p, excite=replace(p.excite, react=out))

    def _react(self, p: Params, kind: int, min_events: int) -> Reaction:
        mu, dt, rows, side, key = self._path(p, kind)
        ev = key[rows, side]
        cells, n = np.unique(ev, return_counts=True)
        cells = cells[n >= min_events]
        G = cells.size + 1
        M = np.bincount(_group(cells, key).ravel(), (mu * dt[:, None]).ravel(), G)
        g_ev = _group(cells, ev)
        mu_ev = mu[rows, side]
        free = np.bincount(g_ev, minlength=G) > 0
        F = int(free.sum())
        X, C = self._kernels(p.excite.beta, rows, side, key >= 0)

        def f(x):
            psi = np.ones(G)
            psi[free] = x[:F]
            lam = psi[g_ev] * mu_ev + X @ x[F:]
            ll = np.log(lam).sum() - psi @ M - C @ x[F:]
            g_psi = np.bincount(g_ev, mu_ev / lam, G) - M
            return -ll, -np.concatenate((g_psi[free], X.T @ (1 / lam) - C))

        x0 = np.concatenate((np.ones(F), np.full(X.shape[1], 0.01)))
        bounds = [(1e-9, None)] * F + [(0.0, None)] * X.shape[1]
        opts = {"ftol": 1e-13, "gtol": 1e-7, "maxiter": 10_000}
        res = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=bounds, options=opts)
        if not res.success:
            raise RuntimeError(f"reaction fit did not converge: {res.message}")
        psi = np.ones(G)
        psi[free] = res.x[:F]
        k = len(p.excite.beta)
        own, cross = (tuple(float(v) for v in res.x[F + j * k : F + (j + 1) * k]) for j in (0, 1))
        scale = {int(c): float(v) for c, v in zip(cells, psi[1:], strict=True)}
        return Reaction(own, cross, float(psi[0]), scale)

    def loglik(self, p: Params, kind: int = MARKET) -> float:
        mu, dt, rows, side, key = self._path(p, kind)
        lam, comp = mu[rows, side], mu.sum(axis=1) @ dt
        own, cross, beta = _gains(p.excite, kind)
        if beta:
            X, C = self._kernels(beta, rows, side, key >= 0)
            w = np.array((*own, *cross))
            lam, comp = lam + X @ w, comp + C @ w
        return float(np.log(lam).sum() - comp)

    def residuals(self, p: Params, kind: int = MARKET) -> list[np.ndarray]:
        mu, dt, rows, side, key = self._path(p, kind)
        src, by = self.t[self.hits[:, 0]], self.hits[:, 1]
        t = self.t[rows]
        big = [np.cumsum(mu[:, s] * dt)[rows[side == s]] for s in Side]
        for a, c, b in zip(*_gains(p.excite, kind), strict=True):
            dead = _dead(src, by, self.t, key >= 0, b)
            for s in Side:
                mine = side == s
                d, mass, _ = dead[s]
                cut = np.searchsorted(d, rows[mine])
                for w, k in ((a, s), (c, 1 - s)):
                    x = src[by == k]
                    n = np.searchsorted(x, t[mine], side="left")
                    lost = np.concatenate(([0.0], np.cumsum(mass[k])))[cut]
                    big[s] += w * (n - _decay(x, t[mine], b)[0] - lost)
        return [np.diff(x, prepend=0.0) for x in big]

    def _kernels(self, beta, rows, side, live) -> tuple[np.ndarray, np.ndarray]:
        src, by = self.t[self.hits[:, 0]], self.hits[:, 1]
        t = self.t[rows]
        k = len(beta)
        X, C = np.zeros((t.size, 2 * k)), np.zeros(2 * k)
        for j, b in enumerate(beta):
            for s in Side:
                mine = side == s
                X[mine, j] = b * _decay(src[by == s], t[mine], b)[0]
                X[mine, k + j] = b * _decay(src[by != s], t[mine], b)[0]
            total = -np.expm1(-b * (self.t[-1] - src)).sum()
            dead = _dead(src, by, self.t, live, b)
            for w in (0, 1):
                C[w * k + j] = total - sum(m[w ^ s].sum() for s, (_, m, _) in enumerate(dead))
        return X, C

    def _path(self, p: Params, kind: int = MARKET):
        if self.top is None or (kind != MARKET and self.quotes is None):
            raise ValueError("this fit has no row data; refit with qr.fit")
        if p.aes != tuple(int(a) for a in self.aes) or p.edges != self.edges:
            raise ValueError("params come from a fit with different aes or edges")
        i, r, n = np.moveaxis(self.top, -1, 0)
        sized = np.array([bool(p.sizes.get((kind, j))) for j in range(p.K)])
        ok = (i >= 0) & sized[np.maximum(i, 0)]
        i = np.where(ok, i, 0)
        mu = np.where(ok, np.array(p.rates)[i, r, n, kind], 0.0)
        key = np.where(ok, _keys(p, i, r, n), -1)
        if kind != MARKET:
            key[mu == 0] = -1
        if p.clock is not None:
            mu *= np.array(p.clock.scale)[self.h, kind][:, None]
        if kind == MARKET:
            rows, side = self.hits.T
        else:
            q = self.quotes[self.quotes[:, 2] == kind]
            rows, side = q[key[q[:, 0], q[:, 1]] >= 0, :2].T
            react = _reaction(p.excite, kind)
            if react is not None:
                mu = mu * _lookup(react, key)
        return mu, np.diff(self.t, prepend=0.0), rows, side, key

    def _back(self) -> tuple[tuple[float, ...], ...]:
        lvl, _, kind, _, _, pos = self.events.T
        out = []
        for i in range(self.aes.size):
            u = pos[(lvl == i) & (kind == CANCEL) & (pos >= 0)] / POSITIONS
            out.append(tuple(float(x) for x in u) or (0.5,))
        return tuple(out)

    def _sizes(self, mode: str, regimes: bool, memory: bool, min_events: int) -> dict:
        lvl, n, kind, qty, regime, _ = self.events.T
        if not memory:
            regime = regime % BASE
        out = {}
        for i in range(self.aes.size):
            for k in (LIMIT, CANCEL, MARKET):
                mine = (lvl == i) & (kind == k)
                if mode == UNIT:
                    out[k, i] = (int(self.aes[i]),)
                    continue
                signed = qty[mine]
                out[k, i] = tuple(int(q) for q in (signed if mode == QUEUE else np.abs(signed)))
                if mode == QUEUE:
                    for b in np.unique(n[mine]):
                        cell = signed[n[mine] == b]
                        if cell.size >= min_events:
                            out[k, i, int(b)] = tuple(int(q) for q in cell)
                        for r in np.unique(regime[mine]) if regimes else ():
                            cell = signed[(n[mine] == b) & (regime[mine] == r)]
                            if cell.size >= min_events:
                                out[k, i, int(b), int(r)] = tuple(int(q) for q in cell)
        return out

    def invariant(self, i: int) -> np.ndarray:
        c, t = self.counts[i].sum(axis=0), self.tau[i].sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(t[:, None] > 0, c / t[:, None], 0.0)
        return invariant(r[:, LIMIT], r[:, CANCEL] + r[:, MARKET])


def _group(cells: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.where(np.isin(x, cells), np.searchsorted(cells, x) + 1, 0)


def _reaction(e: Excite | None, kind: int) -> Reaction | None:
    return None if e is None or not e.react else e.react.get(kind)


def _gains(e: Excite | None, kind: int) -> tuple[tuple[float, ...], ...]:
    if e is not None and kind == MARKET:
        return e.own, e.cross, e.beta
    r = _reaction(e, kind)
    return ((), (), ()) if r is None else (r.own, r.cross, e.beta)


def _lookup(r: Reaction, key: np.ndarray) -> np.ndarray:
    out = np.full(key.shape, r.rest)
    if r.cells:
        k = np.array(sorted(r.cells), dtype=np.int64)
        v = np.array([r.cells[c] for c in k])
        j = np.minimum(np.searchsorted(k, key), k.size - 1)
        hit = k[j] == key
        out[hit] = v[j[hit]]
    return out


def _ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.zeros_like(b, dtype=float), where=b > 0)


def _scale(arrivals: np.ndarray, expected: np.ndarray, mode: str) -> np.ndarray:
    if mode == SHARED:
        arrivals, expected = (
            arrivals.sum(axis=1, keepdims=True),
            expected.sum(axis=1, keepdims=True),
        )
    return _ratio(arrivals, expected) * np.ones(3)


def _clock(slots: np.ndarray, s: np.ndarray) -> Clock:
    return Clock(tuple(float(e) for e in slots), tuple(tuple(float(x) for x in r) for r in s))


def rescale(p: Params, f: Fit, mode: str = KINDS) -> Clock:
    if p.aes != tuple(int(a) for a in f.aes) or p.edges != f.edges:
        raise ValueError("fit the other window with the same aes and edges as the params")
    expected = np.einsum("irnk,irnh->hk", np.array(p.rates), f.exposure)
    return _clock(f.slots, _scale(f.arrivals, expected, mode))


def _nested(rates: np.ndarray) -> tuple:
    return tuple(tuple(tuple(tuple(map(float, c)) for c in r) for r in lvl) for lvl in rates)


def _keys(p: Params, i, r, n):
    r = r if p.memory else r % BASE if p.regimes else 0 * r
    return (i * REGIMES + r) * (p.N + 1) + n


def _decay(t: np.ndarray, at: np.ndarray, beta: float) -> tuple[np.ndarray, np.ndarray]:
    hi = np.searchsorted(t, at, "left")
    m = int((hi - np.searchsorted(t, at - FAR / beta, "left")).max(initial=0))
    if t.size and at.size and 8 * m > np.unique(np.floor(np.r_[t, at] * beta / 300)).size:
        return decayed_at(t, at, beta)
    r, q = np.zeros(at.size), np.zeros(at.size)
    for lag in range(1, m + 1):
        j = hi - lag
        ok = j >= 0
        d = at[ok] - t[j[ok]]
        e = np.exp(-beta * d)
        r[ok] += e
        q[ok] += d * e
    return r, q


def _dead(t, side, at, live, beta):
    out = []
    for s in Side:
        d = np.flatnonzero(~live[:, s])
        a = np.where(d > 0, at[np.maximum(d - 1, 0)], 0.0)
        mass, slope = np.zeros((2, d.size)), np.zeros((2, d.size))
        for src in Side:
            x = t[side == src]
            ra, qa = _decay(x, a, beta)
            rb, qb = _decay(x, at[d], beta)
            tie = np.searchsorted(x, a, "right") - np.searchsorted(x, a, "left")
            gap = at[d] > a
            mass[src], slope[src] = gap * (ra + tie - rb), gap * (qb - qa)
        out.append((d, mass, slope))
    return out


def _excite_score(psi, th, M, g_ev, mu, t, side, at, live):
    k = th.size // 3
    own, cross, beta = th[:k], th[k : 2 * k], th[2 * k :]
    terms = []
    lam = psi[g_ev] * mu
    for j in range(k):
        r = np.zeros((2, t.size))
        q = np.zeros((2, t.size))
        for s in Side:
            mine = side == s
            for w, src in ((0, mine), (1, ~mine)):
                r[w, mine], q[w, mine] = _decay(t[src], t[mine], beta[j])
        terms.append((r, q))
        lam = lam + beta[j] * (own[j] * r[0] + cross[j] * r[1])
    inv = 1 / lam
    ll = np.log(lam).sum() - psi @ M
    g_psi = np.bincount(g_ev, mu * inv, psi.size) - M
    g = np.zeros(th.size)
    for j, (r, q) in enumerate(terms):
        x = at[-1] - t
        total, h = -np.expm1(-beta[j] * x).sum(), x @ np.exp(-beta[j] * x)
        dead = _dead(t, side, at, live, beta[j])
        D = [total - sum(m[w ^ s].sum() for s, (_, m, _) in enumerate(dead)) for w in (0, 1)]
        dD = [h - sum(q_[w ^ s].sum() for s, (_, _, q_) in enumerate(dead)) for w in (0, 1)]
        ll -= own[j] * D[0] + cross[j] * D[1]
        g[j] = beta[j] * (r[0] @ inv) - D[0]
        g[k + j] = beta[j] * (r[1] @ inv) - D[1]
        d = own[j] * (r[0] - beta[j] * q[0]) + cross[j] * (r[1] - beta[j] * q[1])
        g[2 * k + j] = d @ inv - own[j] * dD[0] - cross[j] * dD[1]
    return float(ll), g_psi, g


def _filled(n, t, pooled_n, pooled_t, min_events):
    rate = np.zeros_like(n, dtype=float)
    last = None
    for b in range(n.shape[0]):
        if n[b].sum() >= min_events:
            last = n[b] / t[b]
        elif pooled_n[b].sum() >= min_events:
            last = pooled_n[b] / pooled_t[b]
        if last is not None:
            rate[b] = last
    rate[-1, LIMIT] = 0.0
    return tuple(tuple(float(x) for x in row) for row in rate)


def invariant(birth: np.ndarray, death: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(death[1:] > 0, birth[:-1] / death[1:], 0.0)
    pi = np.concatenate(([1.0], np.cumprod(ratio)))
    return pi / pi.sum()


def fit(
    msgs: Sequence[Msg],
    start: float,
    end: float,
    *,
    depth: int = 5,
    bins: int = 30,
    tick: int = 100,
    slot: float = 30.0,
    aes: Sequence[int] | None = None,
    edges: tuple[int, int] | None = None,
) -> Fit:
    book = ReplayBook()
    gap = Gap()
    lo, hi = start * NS_PER_MINUTE, end * NS_PER_MINUTE
    rows, codes, ups, ts, events = [], [], [], [], []

    for batch in batches(list(on_grid(msgs, tick))):
        t = batch[0].t
        if t >= hi:
            break
        bid, ask = book.best_bid(), book.best_ask()
        market = batch[0].kind in "EC"
        if t >= lo and bid is not None and ask is not None:
            row = len(rows)
            rows.append(_ladder(book, bid, ask, depth))
            codes.append(
                (gap.regime(Side.BUY), gap.regime(Side.SELL)) if ask - bid > 1 else (-1, -1)
            )
            ups.append(gap.up)
            ts.append(t)
            events.extend(_events(book, batch, bid, ask, row, depth))
        for m in batch:
            book.apply(m, (0, 0))
        bid, ask = book.best_bid(), book.best_ask()
        if bid is not None and ask is not None:
            gap.update(bid, ask, market)

    q = np.array(rows, dtype=np.int64)
    ev = np.array(events, dtype=np.int64).reshape(-1, 6)
    if aes is None:
        aes = [max(1, round(np.abs(ev[ev[:, 2] == i, 4]).mean())) for i in range(depth)]
    aes = np.array(aes, dtype=np.int64)
    dt = np.diff(np.array([lo, *ts], dtype=float)) / NS_PER_MINUTE
    if edges is None:
        touch = q[:, :, 0].ravel()
        edges = tuple(int(v) for v in np.quantile(touch[touch > 0], [1 / 3, 2 / 3]))
    low, high = edges
    H = max(1, math.ceil((end - start) / slot - 1e-9))
    slots = np.minimum(start + slot * np.arange(H + 1), end)
    h = np.minimum((np.array(ts) - lo) // (slot * NS_PER_MINUTE), H - 1).astype(np.int64)
    n = np.minimum(-(-q // aes), bins)
    regime = _regimes(q, np.array(codes, dtype=np.int64), np.array(ups), low, high)

    counts = np.zeros((depth, REGIMES, bins + 1, 3), dtype=np.int64)
    exposure = np.zeros((depth, REGIMES, bins + 1, H))
    for s in Side:
        for i in range(depth):
            np.add.at(exposure, (i, regime[:, s, i], n[:, s, i], h), dt)
    row, side, lvl, kind, qty, pos = ev.T
    b = n[row, side, lvl]
    np.add.at(counts, (lvl, regime[row, side, lvl], b, kind), 1)
    arrivals = np.zeros((H, 3), dtype=np.int64)
    np.add.at(arrivals, (h[row], kind), 1)
    has = q > 0
    i = np.where(has.any(axis=2), has.argmax(axis=2), -1)
    at = np.arange(len(q))[:, None], np.arange(2)[None, :], np.maximum(i, 0)
    top = np.stack((i, regime[at], n[at]), axis=-1)
    hits = ev[kind == MARKET][:, :2]
    near = (kind != MARKET) & (lvl == i[row, side])
    grid = np.searchsorted(np.cumsum(dt), np.arange(1, int((end - start) * 60)) / 60)
    sample = q[np.minimum(grid, len(q) - 1)]
    return Fit(
        end - start,
        counts,
        exposure.sum(axis=-1),
        aes,
        (low, high),
        np.column_stack((lvl, b, kind, qty, regime[row, side, lvl], pos)),
        np.concatenate((sample[:, 0], sample[:, 1])),
        slots,
        exposure,
        arrivals,
        (np.array(ts, dtype=float) - lo) / NS_PER_MINUTE,
        h,
        top,
        hits,
        np.column_stack((row, side, kind))[near],
    )


def _ladder(book: Book, bid: int, ask: int, depth: int) -> tuple:
    out = []
    for side, px, step in ((Side.BUY, ask, -1), (Side.SELL, bid, 1)):
        out.append(
            [
                lvl.qty if (lvl := book.level(side, px + step * i)) is not None else 0
                for i in range(1, depth + 1)
            ]
        )
    return out


def _events(book: Book, batch, bid: int, ask: int, row: int, depth: int) -> Iterator[tuple]:
    def level(side: Side, px: int) -> int | None:
        i = ask - px if side == Side.BUY else px - bid
        return i - 1 if 1 <= i <= depth else None

    hit = [0, 0]
    for m in batch:
        match m.kind:
            case "A" | "F":
                if (i := level(m.side, m.px)) is not None:
                    yield row, m.side, i, LIMIT, m.qty, -1
            case "E" | "C":
                hit[book.find(-m.ref).side] += m.qty
            case "X" | "D":
                o = book.find(-m.ref)
                if (i := level(o.side, o.price)) is not None:
                    qty = m.qty if m.kind == "X" else o.qty
                    yield row, o.side, i, CANCEL, *_cancel(book, o, qty)
            case "U":
                o = book.find(-m.ref)
                if (i := level(o.side, o.price)) is not None:
                    yield row, o.side, i, CANCEL, *_cancel(book, o, o.qty)
                if (i := level(o.side, m.px)) is not None:
                    yield row, o.side, i, LIMIT, m.qty, -1
    for side in Side:
        if hit[side]:
            px = bid if side == Side.BUY else ask
            if (i := level(side, px)) is not None:
                gone = hit[side] >= book.level(side, px).qty
                yield row, side, i, MARKET, -hit[side] if gone else hit[side], -1


def _cancel(book: Book, o: Order, qty: int) -> tuple[int, int]:
    q = book.level(o.side, o.price).qty
    pos = -1 if q == o.qty else POSITIONS * book.ahead(o.id) // (q - o.qty)
    return -qty if qty >= q else qty, pos


def _regimes(
    q: np.ndarray, codes: np.ndarray, ups: np.ndarray, low: int, high: int
) -> np.ndarray:
    out = np.zeros(q.shape, dtype=np.int64)
    for s in Side:
        opp = q[:, 1 - s, 0]
        out[:, s, 0] = np.select([opp <= low, opp <= high], [THIN, USUAL], DEEP)
        wide = codes[:, s] >= 0
        out[wide, s, :] = codes[wide, s, None]
        out[ups == (s == Side.BUY), s, :] += BASE
    return out


class QueueReactive:
    def __init__(self, params: Params, owner: int = 0, seed: int = 0, start: float = 0.0) -> None:
        self.params = params
        self.owner = owner
        self.t = start
        self.gap = Gap()
        self._rng = random.Random(seed)
        self._bid: int | None = None
        self._ask: int | None = None
        self._market = False
        self._far = [0, 0]
        e = params.excite
        k = 0 if e is None else len(e.beta)
        self._kicks = [[0.0, 0.0] for _ in range(k)]
        self._kicked = start
        self._react = {} if e is None or not e.react else e.react
        self._gains = [
            [(MARKET, e.own[j], e.cross[j])]
            + [(kind, r.own[j], r.cross[j]) for kind, r in self._react.items()]
            for j in range(k)
        ]

    def seed(self, book: Book, bid: int, ask: int, ids: Iterator[int]) -> None:
        for side, top, step in ((Side.BUY, bid, -1), (Side.SELL, ask, 1)):
            for i in range(1, self.params.K + 1):
                for qty in self._lump(i, self._rng):
                    book.submit(next(ids), side, top + step * (i - 1), qty, owner=self.owner)
            self._far[side] = top + step * (self.params.K - 1)

    def step(
        self, book: Book, rng: random.Random, horizon: float, ids: Iterator[int]
    ) -> tuple[float, Callable[[Book], list[Event]]] | None:
        bid, ask = self._touch(book)
        exposed = self._exposed(bid, ask, rng, ids)
        if exposed:
            return 0.0, lambda b: [e for o, s, px, q in exposed for e in self._add(b, o, s, px, q)]
        actions, weights, hot = self._menu(book, bid, ask)
        clock = self.params.clock
        waited = 0.0
        while True:
            left = horizon - waited
            if clock is None:
                edge, w = math.inf, weights
            else:
                h, edge = clock.slot(self.t)
                m = clock.scale[h]
                w = [x * m[a[0]] for a, x in zip(actions, weights, strict=True)]
            x = self._excess(self.t)
            extra = sum(x[k][s] for _, k, s in hot)
            cum = list(accumulate(_kicked(w, hot, x)))
            dt = rng.expovariate(cum[-1]) if cum and cum[-1] > 0 else math.inf
            if dt <= min(left, edge):
                self.t += dt
                if extra:
                    bound = cum[-1]
                    cum = list(accumulate(_kicked(w, hot, self._excess(self.t))))
                    if rng.random() * bound >= cum[-1]:
                        waited += dt
                        continue
                k = bisect_right(cum, rng.random() * cum[-1], 0, len(cum) - 1)
                return waited + dt, self._fire(book, actions[k], rng, ids)
            if left <= edge:
                self.t += left
                return None
            waited += edge
            self.t += edge

    def _excess(self, t: float) -> list[list[float]]:
        out = [[0.0, 0.0] for _ in range(3)]
        if not self._kicks:
            return out
        e = self.params.excite
        for b, (r0, r1), gains in zip(e.beta, self._kicks, self._gains, strict=True):
            f = b * math.exp(-b * (t - self._kicked))
            u0, u1 = f * r0, f * r1
            for kind, a, c in gains:
                x = out[kind]
                x[0] += a * u0 + c * u1
                x[1] += a * u1 + c * u0
        return out

    def _kick(self, side: Side) -> None:
        for b, r in zip(self.params.excite.beta, self._kicks, strict=True):
            f = math.exp(-b * (self.t - self._kicked))
            r[0] *= f
            r[1] *= f
            r[side] += 1.0
        self._kicked = self.t

    def _add(self, b: Book, oid: int, side: Side, px: int, qty: int) -> list[Event]:
        return b.submit(oid, side, px, qty, owner=self.owner)

    def _touch(self, book: Book) -> tuple[int, int]:
        bid, ask = book.best_bid(), book.best_ask()
        if bid is not None:
            self._bid = bid
        if ask is not None:
            self._ask = ask
        if self._bid is None or self._ask is None:
            raise ValueError("the book needs both a bid and an ask once before flow can start")
        self.gap.update(self._bid, self._ask, self._market)
        return self._bid, self._ask

    def _exposed(self, bid: int, ask: int, rng: random.Random, ids: Iterator[int]) -> list[tuple]:
        K = self.params.K
        out = []
        reach = ask - K
        while self._far[Side.BUY] > reach:
            self._far[Side.BUY] -= 1
            px = self._far[Side.BUY]
            out.extend((next(ids), Side.BUY, px, q) for q in self._lump(K, rng))
        reach = bid + K
        while self._far[Side.SELL] < reach:
            self._far[Side.SELL] += 1
            px = self._far[Side.SELL]
            out.extend((next(ids), Side.SELL, px, q) for q in self._lump(K, rng))
        return out

    def _lump(self, i: int, rng: random.Random) -> list[int]:
        p = self.params
        left = rng.choice(p.queues[i - 1])
        sizes = p.sizes[LIMIT, i - 1]
        out = []
        while left > 0:
            q = min(rng.choice(sizes), left)
            out.append(q)
            left -= q
        return out

    def _menu(self, book: Book, bid: int, ask: int) -> tuple[list[tuple], list[float], list]:
        p = self.params
        gap = self.gap
        low, high = p.edges
        wide = ask - bid > 1
        actions: list[tuple] = []
        weights: list[float] = []
        hot: list[tuple[int, int, Side]] = []
        for side, far, best, step in ((Side.BUY, ask, bid, -1), (Side.SELL, bid, ask, 1)):
            shift = BASE if p.memory and gap.up == (side == Side.BUY) else 0
            if wide:
                r0 = rest = gap.regime(side) + shift
            else:
                opp = book.level(side.opposite, far)
                q = opp.qty if opp is not None else 0
                r0 = (THIN if q <= low else USUAL if q <= high else DEEP) + shift
                rest = shift
            for i in range(p.K):
                px = far + step * (i + 1)
                lvl = book.level(side, px)
                q = lvl.qty if lvl is not None else 0
                a = p.aes[i]
                n = min((q + a - 1) // a, p.N)
                r = rest if i else r0
                lam_l, lam_c, lam_m = p.rates[i][r][n]
                touch = q and px == best
                for kind, lam in ((LIMIT, lam_l), (CANCEL, lam_c)):
                    if not lam or (kind == CANCEL and not q):
                        continue
                    if touch and kind in self._react and p.sizes.get((kind, i)):
                        react = self._react[kind]
                        key = (i * REGIMES + (r if p.regimes else 0)) * (p.N + 1) + n
                        lam *= react.cells.get(key, react.rest)
                        hot.append((len(actions), kind, side))
                    actions.append((kind, side, i, n, r, px))
                    weights.append(lam)
                if touch and (lam_m or (self._kicks and p.sizes.get((MARKET, i)))):
                    if self._kicks:
                        hot.append((len(actions), MARKET, side))
                    actions.append((MARKET, side, i, n, r, px))
                    weights.append(lam_m)
        return actions, weights, hot

    def _fire(self, book: Book, action: tuple, rng: random.Random, ids: Iterator[int]):
        kind, side, i, n, r, px = action
        self._market = kind == MARKET
        sizes = self.params.sizes
        s = rng.choice(sizes.get((kind, i, n, r)) or sizes.get((kind, i, n)) or sizes[kind, i])
        if kind == LIMIT:
            oid = next(ids)
            return lambda b: self._add(b, oid, side, px, s)
        q = sum(o.qty for o in book.level(side, px).orders.values() if o.owner == self.owner)
        if q == 0:
            return lambda b: []
        qty = self._take(s, q)
        if kind == CANCEL:
            return self._pull(book.level(side, px), i, qty, rng)
        if self._kicks:
            self._kick(side)
        oid, taker = next(ids), side.opposite
        return lambda b: b.submit(oid, taker, 0, qty, owner=self.owner, kind=OrderType.MARKET)

    def _take(self, s: int, q: int) -> int:
        if not self.params.conditioned:
            return s
        return max(q, -s) if s < 0 else max(min(s, q - 1), 1)

    def _pull(self, lvl: Level, i: int, qty: int, rng: random.Random):
        mine = [o for o in lvl.orders.values() if o.owner == self.owner]
        back = self.params.back
        u = rng.random() if back is None else rng.choice(back[i])
        target = u * sum(o.qty for o in mine)
        ahead = j = 0
        while j < len(mine) - 1 and ahead + mine[j].qty <= target:
            ahead += mine[j].qty
            j += 1
        plan = []
        for o in mine[j:] + mine[:j][::-1]:
            if qty <= 0:
                break
            left = o.qty - qty
            plan.append((o.id, o.price, max(left, 0)))
            qty -= o.qty
        return lambda b: [e for oid, px, left in plan for e in _shrink(b, oid, px, left)]


def _kicked(w: list[float], hot: list[tuple[int, int, Side]], x: list[list[float]]) -> list[float]:
    if not hot:
        return w
    w = list(w)
    for j, kind, s in hot:
        w[j] += x[kind][s]
    return w


def _shrink(b: Book, oid: int, px: int, left: int) -> list[Event]:
    return b.modify(oid, px, left) if left else b.cancel(oid)


def messages(
    p: Params, minutes: float, seed: int = 0, bid: int = 100, tick: int = 100
) -> list[Msg]:
    book, rng, ids = Book(), random.Random(seed), count(1)
    flow = QueueReactive(p, seed=seed)
    flow.seed(book, bid, bid + 1, ids)
    resting = [o for s in Side for o in book.orders(s)]
    out = [Msg(0, "A", o.id, o.side, o.price * tick, o.qty) for o in resting]
    left = {o.id: o.qty for o in resting}
    t = 0.0
    while (step := flow.step(book, rng, minutes - t, ids)) is not None:
        t += step[0]
        ns = round(t * NS_PER_MINUTE)
        for e in step[1](book):
            if e.type == EventType.TRADE:
                out.append(Msg(ns, "E", e.maker, qty=e.qty))
                left[e.maker] -= e.qty
            elif e.type == EventType.CANCELED:
                out.append(Msg(ns, "D", e.id))
            elif e.type == EventType.MODIFIED:
                out.append(Msg(ns, "X", e.id, qty=left[e.id] - e.qty))
                left[e.id] = e.qty
            elif e.type == EventType.ACCEPTED and e.id in book:
                out.append(Msg(ns, "A", e.id, e.side, e.price * tick, e.qty))
                left[e.id] = e.qty
    return out
