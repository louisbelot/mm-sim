from __future__ import annotations

from dataclasses import replace
from typing import NamedTuple

import numpy as np

from .duopoly import Game, advance, depth, equilibrium, gains, joint, side


class Stationary(NamedTuple):
    V1: np.ndarray
    V2: np.ndarray
    quotes: tuple[np.ndarray, ...]
    eps: float = 0.0


def levels(game: Game) -> np.ndarray:
    return depth(game, np.arange(game.J + 1))


def shape(game: Game) -> tuple[int, int]:
    return 2 * game.Q[0] + 1, 2 * game.Q[1] + 1


def value(game: Game, gamma: float, quotes) -> tuple[np.ndarray, np.ndarray]:
    m = shape(game)
    n = m[0] * m[1]
    zero = np.zeros(m)
    r1, r2 = advance(game, zero, zero, quotes)
    M1, M2 = np.empty((n, n)), np.empty((n, n))
    for i, e in enumerate(np.eye(n).reshape(n, *m)):
        W1, W2 = advance(game, gamma * e, gamma * e, quotes)
        M1[:, i], M2[:, i] = (W1 - r1).ravel(), (W2 - r2).ravel()
    eye = np.eye(n)
    V1 = np.linalg.solve(eye - M1, r1.ravel())
    V2 = np.linalg.solve(eye - M2, r2.ravel())
    return V1.reshape(m), V2.reshape(m)


def deviation(game: Game, V1, V2, quotes) -> tuple[np.ndarray, ...]:
    q1, q2 = game.grid()
    Q1, Q2 = game.Q
    d = depth(game, np.arange(game.J + 2))[:, None, None]
    off = np.isinf(d)
    b1, a1, b2, a2 = quotes
    up1, up2, dn1, dn2 = gains(V1)
    vp1, vp2, vd1, vd2 = gains(V2)
    out = []
    for x1, x2, g, c1, c2 in (
        (b1, b2, (up1, up2, vp2, vp1), q1 < Q1, q2 < Q2),
        (a1, a2, (dn1, dn2, vd2, vd1), q1 > -Q1, q2 > -Q2),
    ):
        u1, u2 = side(game, x1, x2, *g)
        best1 = np.where(off | c1, side(game, d, x2, *g)[0], -np.inf).max(axis=0)
        best2 = np.where(off | c2, side(game, x1, d, *g)[1], -np.inf).max(axis=0)
        out += [best1 - u1, best2 - u2]
    return out[0], out[2], out[1], out[3]


def gap(game: Game, gamma: float, quotes, V1, V2) -> float:
    return max(x.max() for x in deviation(game, gamma * V1, gamma * V2, quotes))


def exploit(game: Game, gamma: float, quotes, V1, V2) -> float:
    B1 = respond(game, gamma, quotes, 1)[0]
    B2 = respond(game, gamma, quotes, 2)[0]
    return max((B1 - V1).max(), (B2 - V2).max())


def polish(game: Game, gamma: float, quotes, tol: float, rounds: int) -> Stationary | None:
    seen = set()
    for _ in range(rounds):
        V1, V2 = value(game, gamma, quotes)
        W1, W2 = gamma * V1, gamma * V2
        gb1, ga1, gb2, ga2 = deviation(game, W1, W2, quotes)
        if max(gb1.max(), ga1.max(), gb2.max(), ga2.max()) <= tol:
            return Stationary(V1, V2, quotes)
        key = np.array(quotes).tobytes()
        if key in seen:
            return None
        seen.add(key)
        fresh, _ = equilibrium(game, W1, W2)
        keep_b = (gb1 <= tol) & (gb2 <= tol)
        keep_a = (ga1 <= tol) & (ga2 <= tol)
        quotes = tuple(
            np.where(keep, old, new)
            for old, new, keep in zip(quotes, fresh, (keep_b, keep_a, keep_b, keep_a), strict=True)
        )
    return None


def nearest(game: Game, gamma: float, tol: float, rounds: int, horizon: int, tries: int):
    V1 = V2 = np.zeros(shape(game))
    best = None
    for i in range(horizon):
        quotes, _ = equilibrium(game, gamma * V1, gamma * V2)
        V1, V2 = advance(game, gamma * V1, gamma * V2, quotes)
        if (horizon - 1 - i) % 25 or i < horizon - 25 * tries:
            continue
        sol = polish(game, gamma, quotes, tol, rounds)
        if sol is not None:
            return sol
        E1, E2 = value(game, gamma, quotes)
        g = gap(game, gamma, quotes, E1, E2)
        if best is None or g < best[0]:
            best = g, E1, E2, quotes
    _, E1, E2, quotes = best
    return Stationary(E1, E2, quotes, exploit(game, gamma, quotes, E1, E2))


def markov(
    game: Game,
    gamma: float,
    steps: int = 20,
    tol: float = 1e-10,
    rounds: int = 200,
    horizon: int = 1500,
    tries: int = 20,
) -> Stationary:
    if game.tick <= 0:
        raise ValueError("competition needs a tick")
    zero = np.zeros(shape(game))
    quotes, _ = equilibrium(game, zero, zero)
    for g in np.linspace(0, gamma, steps + 1)[1:]:
        sol = polish(game, g, quotes, tol, rounds)
        if sol is None:
            return nearest(game, gamma, tol, rounds // 4, horizon, tries)
        quotes = sol.quotes
    return sol


def collusive(game: Game, gamma: float, tol: float = 1e-11, cap: int = 200_000) -> Stationary:
    S = np.zeros(shape(game))
    for _ in range(cap):
        quotes, W = joint(game, gamma * S)
        if np.abs(W - S).max() < tol:
            return Stationary(*value(game, gamma, quotes), quotes)
        S = W
    raise RuntimeError(f"no fixed point in {cap} iterations")


def respond(game: Game, gamma: float, quotes, who: int = 1, tol: float = 1e-11, cap: int = 200_000):
    b1, a1, b2, a2 = quotes
    if who == 2:
        game = replace(game, Q=game.Q[::-1], share=1 - game.share)
        b2, a2 = b1.T, a1.T
    q1, q2 = game.grid()
    Q1 = game.Q[0]
    d = depth(game, np.arange(game.J + 2))[:, None, None]
    off = np.isinf(d)
    carry = game.drift(q1 + q2) * game.dt * q1 - game.phi * q1**2 * game.dt
    V = np.zeros(shape(game))
    for _ in range(cap):
        up1, up2, dn1, dn2 = gains(gamma * V)
        ub = np.where(off | (q1 < Q1), side(game, d, b2, up1, up2, 0.0, 0.0)[0], -np.inf)
        ua = np.where(off | (q1 > -Q1), side(game, d, a2, dn1, dn2, 0.0, 0.0)[0], -np.inf)
        W = gamma * V + carry + ub.max(axis=0) + ua.max(axis=0)
        if np.abs(W - V).max() < tol:
            break
        V = W
    else:
        raise RuntimeError(f"no fixed point in {cap} iterations")
    bid, ask = d[ub.argmax(axis=0), 0, 0], d[ua.argmax(axis=0), 0, 0]
    return (W, bid, ask) if who == 1 else (W.T, bid.T, ask.T)


def prior(game: Game, gamma: float, share: float) -> np.ndarray:
    d = levels(game)
    j = np.arange(d.size)
    odds = ((j[:, None] < j[None, :]) + share * (j[:, None] == j[None, :])).mean(axis=1)
    r = game.A * game.dt * np.exp(-game.k * d) * d * odds
    return (r[:, None] + r[None, :]).ravel() / (1 - gamma)


def wins(d, rival, mine) -> np.ndarray:
    return np.isfinite(d) & ((d < rival) | ((d == rival) & mine))


def digit(rival, own) -> np.ndarray:
    return 1 + (rival > own).astype(np.int64) - (rival < own)


def backup(bid, ask, rival_bid, rival_ask, buy, sell, r, mine, stay, up, down, cost, gamma):
    hit_b = buy & (r >= bid) & wins(bid, rival_bid, mine)
    hit_a = sell & (r >= ask) & wins(ask, rival_ask, mine)
    return (
        np.where(hit_b, bid + gamma * up, np.where(hit_a, ask + gamma * down, gamma * stay)) - cost
    )


class Learned(NamedTuple):
    table1: np.ndarray
    table2: np.ndarray
    half: np.ndarray
    reward1: np.ndarray
    reward2: np.ndarray


def train(
    game: Game,
    gamma: float = 0.99,
    steps: int = 1_000_000,
    runs: int = 16,
    alpha: float = 0.1,
    decay: float = 5e-6,
    sync: bool = False,
    window: int = 10_000,
    seed: int = 0,
    memory: bool = False,
    tau: float | None = None,
) -> Learned:
    rng = np.random.default_rng(seed)
    Q = np.array(game.Q)[:, None]
    d = levels(game)
    M = d.size
    C = 9 if memory else 1
    idx = np.arange(runs)
    tables = [
        np.broadcast_to(prior(game, gamma, s), (runs, 2 * n + 1, C, M * M)).copy()
        for s, n in zip((game.share, 1 - game.share), game.Q, strict=True)
    ]
    tops = [t.max(axis=3) for t in tables]
    counts = [np.zeros(t.shape[: 3 if sync else 4]) for t in tables] if tau else None
    q = np.zeros((2, runs), dtype=np.int64)
    c = np.full((2, runs), C // 2)
    a = game.A * game.dt
    pen = game.phi * game.dt
    windows = steps // window
    half = np.zeros((windows, runs))
    rewards = np.zeros((2, windows, runs))
    grid_b = np.broadcast_to(d[:, None], (M, M)).ravel()
    grid_a = np.broadcast_to(d[None, :], (M, M)).ravel()

    for w in range(windows):
        u, coin = rng.random((2, window, runs))
        explore = rng.random((window, 2, runs))
        reach = rng.exponential(1 / game.k, (window, runs))
        picks = rng.integers(M * M, size=(window, 2, runs))
        seen = np.zeros(runs)
        for t in range(window):
            eps = np.exp(-decay * (w * window + t))
            acts = np.where(
                explore[t] < eps,
                picks[t],
                [tables[i][idx, q[i] + Q[i], c[i]].argmax(axis=1) for i in (0, 1)],
            )
            bids = np.where(q < Q, d[acts // M], np.inf)
            asks = np.where(q > -Q, d[acts % M], np.inf)
            buy = u[t] < a
            sell = (u[t] >= a) & (u[t] < 2 * a)
            r = reach[t]
            first = coin[t] < game.share
            inside_b, inside_a = bids.min(axis=0), asks.min(axis=0)
            both = np.isfinite(inside_b) & np.isfinite(inside_a)
            half[w] += np.where(both, (inside_b + inside_a) / 2, 0.0)
            seen += both
            ahead = np.zeros_like(c)
            if memory:
                ahead[0] = 3 * digit(bids[1], bids[0]) + digit(asks[1], asks[0])
                ahead[1] = C - 1 - ahead[0]
            for i in (0, 1):
                j = 1 - i
                n = game.Q[i]
                s = q[i] + n
                table, top = tables[i], tops[i]
                mine = first if i == 0 else ~first
                moves = (s, np.minimum(s + 1, 2 * n), np.maximum(s - 1, 0))
                cost = pen * q[i] ** 2
                if sync:
                    db = np.where((q[i] < n)[:, None], grid_b, np.inf)
                    da = np.where((q[i] > -n)[:, None], grid_a, np.inf)
                    if memory:
                        nc = 3 * digit(bids[j][:, None], db) + digit(asks[j][:, None], da)
                    else:
                        nc = np.zeros((runs, M * M), dtype=np.int64)
                    nxt = [top[idx[:, None], x[:, None], nc] for x in moves]
                    col = (bids[j], asks[j], buy, sell, r, mine)
                    target = backup(db, da, *(x[:, None] for x in col), *nxt, cost[:, None], gamma)
                    key = (idx, s, c[i])
                else:
                    nxt = [top[idx, x, ahead[i]] for x in moves]
                    target = backup(
                        bids[i], asks[i], bids[j], asks[j], buy, sell, r, mine, *nxt, cost, gamma
                    )
                    key = (idx, s, c[i], acts[i])
                if tau is None:
                    lr = alpha
                else:
                    counts[i][key] += 1
                    lr = alpha * tau / (tau + counts[i][key])
                    lr = lr[:, None] if sync else lr
                table[key] += lr * (target - table[key])
                top[idx, s, c[i]] = table[idx, s, c[i]].max(axis=1)
                fb = buy & (r >= bids[i]) & wins(bids[i], bids[j], mine)
                fa = sell & (r >= asks[i]) & wins(asks[i], asks[j], mine)
                rewards[i, w] += np.where(fb, bids[i], 0.0) + np.where(fa, asks[i], 0.0) - cost
                q[i] += fb.astype(np.int64) - fa
            c = ahead
        half[w] /= np.maximum(seen, 1)
    rewards /= window
    if not memory:
        tables = [t[:, :, 0] for t in tables]
    return Learned(tables[0], tables[1], half, rewards[0], rewards[1])


def greedy(game: Game, table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = levels(game)
    Q = (table.shape[0] - 1) // 2
    q = np.arange(-Q, Q + 1).reshape(-1, *[1] * (table.ndim - 2))
    jb, ja = np.divmod(table.argmax(axis=-1), d.size)
    return np.where(q < Q, d[jb], np.inf), np.where(q > -Q, d[ja], np.inf)


def quotes(game: Game, table1: np.ndarray, table2: np.ndarray) -> tuple[np.ndarray, ...]:
    b1, a1 = greedy(game, table1)
    b2, a2 = greedy(game, table2)
    m = (b1.size, b2.size)
    return tuple(
        np.broadcast_to(x, m) for x in (b1[:, None], a1[:, None], b2[None, :], a2[None, :])
    )


class Score(NamedTuple):
    V1: np.ndarray
    V2: np.ndarray
    gap1: np.ndarray
    gap2: np.ndarray


def score(game: Game, gamma: float, learned: Learned) -> Score:
    Q1, Q2 = game.Q
    rows = []
    for t1, t2 in zip(learned.table1, learned.table2, strict=True):
        qs = quotes(game, t1, t2)
        V1, V2 = value(game, gamma, qs)
        B1 = respond(game, gamma, qs, 1)[0]
        B2 = respond(game, gamma, qs, 2)[0]
        rows.append((V1[Q1, Q2], V2[Q1, Q2], B1[Q1, Q2] - V1[Q1, Q2], B2[Q1, Q2] - V2[Q1, Q2]))
    return Score(*np.array(rows).T)
