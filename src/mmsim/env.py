from __future__ import annotations

from collections.abc import Callable, Sequence

import gymnasium as gym
import numpy as np

from .accounting import exposure
from .book import Side
from .bots import Quoter
from .flow import CSTParams, ZeroIntelligence
from .sim import Agent, Context, Latency, Print, Simulator

DEPTHS: tuple[int | None, ...] = (0, 1, 2, None)
FEATURES = ("position", "time left", "spread", "imbalance", "mid move", "market flow")


def sky() -> ZeroIntelligence:
    return ZeroIntelligence(CSTParams.sky_perfect())


class Puppet(Quoter):
    def __init__(
        self,
        Q: int,
        minutes: float,
        depths: Sequence[int | None] = DEPTHS,
        size: int = 1,
    ) -> None:
        super().__init__(size, Q * size)
        self.Q = Q
        self.minutes = minutes
        self.depths = tuple(depths)
        self.pick: tuple[int | None, int | None] | None = None
        self.close = 0.0
        self._mid: float | None = None
        self._flow = 0

    @property
    def n_actions(self) -> int:
        return len(self.depths) ** 2

    def choose(self, a: int) -> None:
        i, j = divmod(a, len(self.depths))
        self.pick = self.depths[i], self.depths[j]

    def targets(self, ctx: Context) -> tuple[int | None, int | None]:
        bid, ask = ctx.bid, ctx.ask
        if self.pick is None or bid is None or ask is None:
            return None, None
        db, da = self.pick
        return (None if db is None else bid - db), (None if da is None else ask + da)

    def on_trade(self, ctx: Context, p: Print) -> None:
        self._flow += p.qty if p.side == Side.BUY else -p.qty

    def observe(self, ctx: Context) -> np.ndarray:
        bid, ask = ctx.bid, ctx.ask
        spread = imb = move = 0.0
        if bid is not None and ask is not None:
            mid = (bid + ask) / 2
            if self._mid is not None:
                move = float(np.clip(mid - self._mid, -5, 5))
            self._mid = mid
            spread = min(ask - bid, 5) - 1
            b, a = self._others(ctx, Side.BUY, bid), self._others(ctx, Side.SELL, ask)
            imb = (b - a) / (b + a) if b + a else 0.0
        flow = float(np.clip(self._flow / 5, -5, 5))
        self._flow = 0
        q = ctx.position / (self.Q * self.size)
        tau = (self.close - ctx.t) / self.minutes
        return np.array([q, tau, spread, imb, move, flow])

    @staticmethod
    def _others(ctx: Context, side: Side, px: int) -> int:
        lvl = ctx.book.level(side, px)
        own = sum(o.qty for o in ctx.orders.values() if o.side == side and o.price == px)
        return (0 if lvl is None else lvl.qty) - own


class Learned(Puppet):
    def __init__(
        self,
        decide: Callable[[np.ndarray], int],
        Q: int,
        minutes: float,
        dt: float,
        depths: Sequence[int | None] = DEPTHS,
        size: int = 1,
    ) -> None:
        super().__init__(Q, minutes, depths, size)
        self.decide = decide
        self.dt = dt

    def on_start(self, ctx: Context) -> None:
        self.close = ctx.t + self.minutes
        self._tick(ctx)

    def on_timer(self, ctx: Context) -> None:
        self._tick(ctx)

    def _tick(self, ctx: Context) -> None:
        self.choose(self.decide(self.observe(ctx)))
        self.requote(ctx)
        ctx.set_timer(self.dt)


class MarketMakingEnv(gym.Env):
    def __init__(
        self,
        flow: Callable[[], ZeroIntelligence] = sky,
        *,
        minutes: float = 60.0,
        dt: float = 1.0,
        Q: int = 5,
        depths: Sequence[int | None] = DEPTHS,
        phi: float = 0.0,
        terminal: float = 0.0,
        warm: float = 30.0,
        latency: Latency | None = None,
        crowd: Callable[[int], Sequence[Agent]] | None = None,
    ) -> None:
        self.flow = flow
        self.crowd = crowd
        self.minutes = minutes
        self.dt = dt
        self.steps = round(minutes / dt)
        self.Q = Q
        self.depths = tuple(depths)
        self.phi = phi
        self.terminal = terminal
        self.warm = warm
        self.latency = latency
        self.action_space = gym.spaces.Discrete(len(self.depths) ** 2)
        self.observation_space = gym.spaces.Box(-6.0, 6.0, shape=(len(FEATURES),), dtype=np.float64)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(2**31))
        self.puppet = Puppet(self.Q, self.minutes, self.depths)
        lat = {} if self.latency is None else {self.puppet: self.latency}
        others = [] if self.crowd is None else list(self.crowd(seed))
        self.sim = Simulator(self.flow(), [self.puppet, *others], seed=seed, latency=lat)
        self.sim.seed(10_000, 10_001)
        self.sim.warm_up(self.warm)
        self.ctx = self.sim.contexts[0]
        self.puppet.close = self.sim.t + self.minutes
        self.n = 0
        self._value = 0.0
        self._seen = 0
        return self.puppet.observe(self.ctx), {}

    def step(self, action: int):
        ctx, acct = self.ctx, self.ctx.account
        t0, q0 = self.sim.t, acct.position
        self.puppet.choose(int(action))
        if self.n:
            self.puppet.requote(ctx)
        self.sim.run(self.dt)
        self.n += 1
        new = acct.fills[self._seen :]
        self._seen = len(acct.fills)
        value = ctx.pnl()
        reward = value - self._value - self.phi * exposure(new, t0, self.sim.t, q0)
        self._value = value
        done = self.n == self.steps
        if done:
            reward -= self.terminal * acct.position**2
        info = {"pnl": value, "position": acct.position, "fills": len(acct.fills)}
        return self.puppet.observe(ctx), float(reward), done, False, info
