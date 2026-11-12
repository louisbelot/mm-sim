from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import numpy as np

from . import robust as r
from .ppo import DEFAULT, PPO, Config, Map, Policy, Stats, Trajectory, rollout

DELTAS = np.linspace(0.0, 2.0, 9)


def menu(game: r.Game, controls: str) -> r.Market:
    b = np.array([game.b[0], 0.0, game.b[1]]) if "b" in controls else np.array([r.FIXED.b])
    A = np.array(game.A) if "A" in controls else np.array([r.FIXED.A])
    k = np.array(game.k) if "k" in controls else np.array([r.FIXED.k])
    bb, AA, kk = np.meshgrid(b, A, k, indexing="ij")
    return r.Market(bb.ravel(), AA.ravel(), kk.ravel())


class Duel:
    def __init__(
        self,
        game: r.Game,
        side: str,
        other: Policy | None = None,
        controls: str = "bAk",
        deltas: np.ndarray = DELTAS,
        start: int | None = None,
    ) -> None:
        if side not in ("mm", "adv"):
            raise ValueError(f"unknown side {side!r}")
        self.game = game
        self.side = side
        self.other = other
        self.opts = menu(game, controls)
        self.deltas = deltas
        self.start = start

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
        self.rng = np.random.default_rng(seed)
        g = self.game
        if self.start is not None:
            self.n = int(self.rng.integers(0, round(0.95 * g.N) + 1))
            self.q = int(self.rng.integers(-self.start, self.start + 1))
        else:
            self.n, self.q = 0, 0
        self.x, self.z = 0.0, 100.0
        return self.obs(), {}

    def obs(self) -> np.ndarray:
        return np.array([1 - self.n / self.game.N, self.q / self.game.Q])

    def _quotes(self, a: int) -> tuple[float, float]:
        i, j = divmod(a, len(self.deltas))
        return float(self.deltas[i]), float(self.deltas[j])

    def _market(self, a: int) -> tuple[float, float, float]:
        return tuple(float(x[a]) for x in self.opts)

    def _other(self, o: np.ndarray) -> int | None:
        if self.other is None:
            return None
        return self.other.act(o, self.rng)[0]

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        g, o = self.game, self.obs()
        other = self._other(o)
        if self.side == "mm":
            db, da = self._quotes(action)
            b, A, k = r.FIXED if other is None else self._market(other)
        else:
            db, da = self._quotes(other)
            b, A, k = self._market(action)
        a = A * g.dt
        pb = 0.0 if self.q == g.Q else a * np.exp(-k * db)
        pa = 0.0 if self.q == -g.Q else a * np.exp(-k * da)
        before = self.x + self.q * self.z
        if self.rng.random() < pb:
            self.x -= self.z - db
            self.q += 1
        if self.rng.random() < pa:
            self.x += self.z + da
            self.q -= 1
        self.z += b * g.dt + g.sigma * np.sqrt(g.dt) * self.rng.standard_normal()
        self.n += 1
        reward = self.x + self.q * self.z - before - float(g.penalty(self.q))
        done = self.n == g.N
        if done:
            reward -= g.eta * self.q**2
        sign = 1.0 if self.side == "mm" else -1.0
        return self.obs(), sign * reward, done, False, {"q": self.q}


def grid(game: r.Game) -> np.ndarray:
    n, q = np.meshgrid(np.arange(game.N), game.q, indexing="ij")
    return np.stack([1 - n / game.N, q / game.Q], axis=-1)


def quotes(policy: Policy, game: r.Game, deltas: np.ndarray = DELTAS):
    a = policy.pi(grid(game)).argmax(axis=-1)
    i, j = np.divmod(a, len(deltas))
    return deltas[i], deltas[j]


def market(policy: Policy, game: r.Game, controls: str = "bAk") -> r.Market:
    a = policy.pi(grid(game)).argmax(axis=-1)
    return r.Market(*(x[a] for x in menu(game, controls)))


def _episode(args: tuple[Callable[[], Duel], Policy, int]) -> Trajectory:
    make, policy, seed = args
    return rollout(make(), policy, seed)


class Log(NamedTuple):
    round: int
    side: str
    reward: float
    stats: Stats


def rarl(
    game: r.Game,
    rounds: int,
    controls: str = "bAk",
    mm_iters: int = 10,
    adv_iters: int = 10,
    cfg: Config = DEFAULT,
    seed: int = 0,
    start: int | None = None,
    pmap: Map = map,
    log: Callable[[Log], None] | None = None,
) -> tuple[PPO, PPO | None]:
    n_mm = len(DELTAS) ** 2
    mm = PPO(2, n_mm, cfg, seed)
    adv = PPO(2, len(menu(game, controls)[0]), cfg, seed + 1) if controls else None
    base = 1_000_000 * (seed + 1)
    total = rounds * (mm_iters + (adv_iters if adv else 0))
    done = 0

    def train(agent: PPO, make: Callable[[], Duel], iters: int, rnd: int, side: str) -> None:
        nonlocal done
        for _ in range(iters):
            seeds = range(base + done * cfg.episodes, base + (done + 1) * cfg.episodes)
            trajs = list(pmap(_episode, [(make, agent.policy, s) for s in seeds]))
            stats = agent.update(trajs, 1 - done / total if cfg.anneal else 1.0)
            done += 1
            if log is not None:
                log(Log(rnd, side, float(np.mean([t.rew.sum() for t in trajs])), stats))

    duel = partial(Duel, game, controls=controls, start=start)
    for rnd in range(rounds):
        train(mm, partial(duel, "mm", None if adv is None else adv.policy), mm_iters, rnd, "mm")
        if adv is not None:
            train(adv, partial(duel, "adv", mm.policy), adv_iters, rnd, "adv")
    return mm, adv
