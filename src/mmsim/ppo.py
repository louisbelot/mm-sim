from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import NamedTuple, Protocol

import numpy as np

from .nn import MLP, Adam, clip_norm


@dataclass(frozen=True)
class Config:
    gamma: float = 1.0
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    batch: int = 256
    lr: float = 3e-4
    entropy: float = 0.01
    max_grad: float = 0.5
    hidden: tuple[int, ...] = (64, 64)
    episodes: int = 16
    anneal: bool = True


DEFAULT = Config()


class Env(Protocol):
    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]: ...

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]: ...


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


class Policy:
    def __init__(self, n_obs: int, n_act: int, hidden: tuple[int, ...], rng: np.random.Generator):
        self.pi = MLP([n_obs, *hidden, n_act], rng, head=0.01)
        self.v = MLP([n_obs, *hidden, 1], rng, head=1.0)

    def probs(self, obs: np.ndarray) -> np.ndarray:
        return np.exp(log_softmax(self.pi(obs)))

    def value(self, obs: np.ndarray) -> float:
        return float(self.v(obs)[0])

    def act(self, obs: np.ndarray, rng: np.random.Generator) -> tuple[int, float, float]:
        lp = log_softmax(self.pi(obs))
        a = int(rng.choice(len(lp), p=np.exp(lp)))
        return a, float(lp[a]), self.value(obs)

    def greedy(self, obs: np.ndarray) -> int:
        return int(np.argmax(self.pi(obs)))


class Trajectory(NamedTuple):
    obs: np.ndarray
    act: np.ndarray
    logp: np.ndarray
    val: np.ndarray
    rew: np.ndarray
    last: float


def rollout(env: Env, policy: Policy, seed: int) -> Trajectory:
    rng = np.random.default_rng(seed)
    o, _ = env.reset(seed=seed)
    obs, act, logp, val, rew = [], [], [], [], []
    done = trunc = False
    while not (done or trunc):
        a, lp, v = policy.act(o, rng)
        obs.append(o)
        act.append(a)
        logp.append(lp)
        val.append(v)
        o, r, done, trunc, _ = env.step(a)
        rew.append(r)
    last = 0.0 if done else policy.value(o)
    return Trajectory(
        np.array(obs), np.array(act), np.array(logp), np.array(val), np.array(rew), last
    )


def gae(
    rew: np.ndarray, val: np.ndarray, last: float, gamma: float, lam: float
) -> tuple[np.ndarray, np.ndarray]:
    nxt = np.append(val[1:], last)
    delta = rew + gamma * nxt - val
    adv = np.empty_like(delta)
    acc = 0.0
    for t in range(len(delta) - 1, -1, -1):
        acc = delta[t] + gamma * lam * acc
        adv[t] = acc
    return adv, adv + val


class Stats(NamedTuple):
    policy: float
    value: float
    entropy: float
    kl: float
    clipped: float


def gradients(
    policy: Policy,
    obs: np.ndarray,
    act: np.ndarray,
    logp_old: np.ndarray,
    adv: np.ndarray,
    ret: np.ndarray,
    clip: float,
    entropy: float,
) -> tuple[Stats, list[np.ndarray], list[np.ndarray]]:
    n = len(act)
    z, cache = policy.pi.forward(obs)
    lp_all = log_softmax(z)
    p = np.exp(lp_all)
    log_ratio = lp_all[np.arange(n), act] - logp_old
    ratio = np.exp(log_ratio)
    surr = ratio * adv
    clipped = np.clip(ratio, 1 - clip, 1 + clip) * adv
    live = surr <= clipped
    H = -(p * lp_all).sum(axis=1)

    dz = -p * (live * surr / n)[:, None]
    dz[np.arange(n), act] += live * surr / n
    dz = -dz + entropy / n * p * (lp_all + H[:, None])
    g_pi = policy.pi.backward(cache, dz)

    v, vcache = policy.v.forward(obs)
    err = v[:, 0] - ret
    g_v = policy.v.backward(vcache, err[:, None] / n)

    stats = Stats(
        float(-np.minimum(surr, clipped).mean() - entropy * H.mean()),
        float(0.5 * (err**2).mean()),
        float(H.mean()),
        float((ratio - 1 - log_ratio).mean()),
        float((np.abs(ratio - 1) > clip).mean()),
    )
    return stats, g_pi, g_v


class PPO:
    def __init__(self, n_obs: int, n_act: int, cfg: Config = DEFAULT, seed: int = 0) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.policy = Policy(n_obs, n_act, cfg.hidden, self.rng)
        self.opt_pi = Adam(self.policy.pi.params, cfg.lr)
        self.opt_v = Adam(self.policy.v.params, cfg.lr)

    def update(self, trajs: Iterable[Trajectory], lr_frac: float = 1.0) -> Stats:
        cfg = self.cfg
        trajs = list(trajs)
        parts = [gae(t.rew, t.val, t.last, cfg.gamma, cfg.lam) for t in trajs]
        obs = np.concatenate([t.obs for t in trajs])
        act = np.concatenate([t.act for t in trajs])
        logp = np.concatenate([t.logp for t in trajs])
        adv = np.concatenate([a for a, _ in parts])
        ret = np.concatenate([r for _, r in parts])
        self.opt_pi.lr = self.opt_v.lr = cfg.lr * lr_frac
        pi, v = self.policy.pi, self.policy.v
        seen = []
        for _ in range(cfg.epochs):
            perm = self.rng.permutation(len(act))
            for k in range(0, len(perm), cfg.batch):
                i = perm[k : k + cfg.batch]
                a = adv[i]
                a = (a - a.mean()) / (a.std() + 1e-8)
                stats, g_pi, g_v = gradients(
                    self.policy, obs[i], act[i], logp[i], a, ret[i], cfg.clip, cfg.entropy
                )
                clip_norm(g_pi, cfg.max_grad)
                clip_norm(g_v, cfg.max_grad)
                self.opt_pi.step(pi.params, g_pi)
                self.opt_v.step(v.params, g_v)
                seen.append(stats)
        return Stats(*np.mean(seen, axis=0))


def _episode(args: tuple[Callable[[], Env], Policy, int]) -> Trajectory:
    make, policy, seed = args
    return rollout(make(), policy, seed)


Map = Callable[[Callable, Iterable], Iterator]


def train(
    make: Callable[[], Env],
    n_obs: int,
    n_act: int,
    iterations: int,
    cfg: Config = DEFAULT,
    seed: int = 0,
    pmap: Map = map,
    log: Callable[[int, float, Stats], None] | None = None,
) -> tuple[PPO, list[float]]:
    agent = PPO(n_obs, n_act, cfg, seed)
    curve = []
    base = 1_000_000 * (seed + 1)
    for it in range(iterations):
        seeds = range(base + it * cfg.episodes, base + (it + 1) * cfg.episodes)
        trajs = list(pmap(_episode, [(make, agent.policy, s) for s in seeds]))
        frac = 1 - it / iterations if cfg.anneal else 1.0
        stats = agent.update(trajs, frac)
        curve.append(float(np.mean([t.rew.sum() for t in trajs])))
        if log is not None:
            log(it, curve[-1], stats)
    return agent, curve
