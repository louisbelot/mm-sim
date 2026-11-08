from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def orthogonal(rows: int, cols: int, gain: float, rng: np.random.Generator) -> np.ndarray:
    a = rng.standard_normal((max(rows, cols), min(rows, cols)))
    q, r = np.linalg.qr(a)
    q *= np.sign(np.diag(r))
    if rows < cols:
        q = q.T
    return gain * q[:rows, :cols]


class MLP:
    def __init__(
        self,
        sizes: Sequence[int],
        rng: np.random.Generator,
        gain: float = np.sqrt(2),
        head: float = 1.0,
    ) -> None:
        n = len(sizes) - 1
        self.W = [
            orthogonal(sizes[i], sizes[i + 1], head if i == n - 1 else gain, rng) for i in range(n)
        ]
        self.b = [np.zeros(s) for s in sizes[1:]]

    @property
    def params(self) -> list[np.ndarray]:
        return self.W + self.b

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for W, b in zip(self.W[:-1], self.b[:-1], strict=True):
            x = np.tanh(x @ W + b)
        return x @ self.W[-1] + self.b[-1]

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        acts = [x]
        for W, b in zip(self.W[:-1], self.b[:-1], strict=True):
            acts.append(np.tanh(acts[-1] @ W + b))
        return acts[-1] @ self.W[-1] + self.b[-1], acts

    def backward(self, acts: list[np.ndarray], dout: np.ndarray) -> list[np.ndarray]:
        dW, db = [], []
        g = dout
        for i in range(len(self.W) - 1, -1, -1):
            dW.append(acts[i].T @ g)
            db.append(g.sum(axis=0))
            if i:
                g = (g @ self.W[i].T) * (1 - acts[i] ** 2)
        return dW[::-1] + db[::-1]


class Adam:
    def __init__(
        self,
        params: list[np.ndarray],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-5,
    ) -> None:
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.n = 0

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        self.n += 1
        c1 = 1 - self.b1**self.n
        c2 = 1 - self.b2**self.n
        for p, g, m, v in zip(params, grads, self.m, self.v, strict=True):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            p -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)


def clip_norm(grads: list[np.ndarray], limit: float) -> float:
    norm = float(np.sqrt(sum((g * g).sum() for g in grads)))
    if norm > limit:
        for g in grads:
            g *= limit / norm
    return norm
