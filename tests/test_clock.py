import math

import numpy as np
import pytest

from mmsim import qr
from mmsim.qr import SHARED, Clock


def test_slot_clamps_at_both_ends():
    c = Clock((0.0, 10.0, 20.0, 30.0), ((1, 1, 1),) * 3)
    assert c.slot(-5.0) == (0, 15.0)
    assert c.slot(5.0) == (0, 5.0)
    assert c.slot(10.0) == (1, 10.0)
    assert c.slot(25.0) == (2, math.inf)
    assert c.slot(40.0) == (2, math.inf)


def synthetic(seed, X=20, H=5):
    rng = np.random.default_rng(seed)
    lam = rng.uniform(5, 50, (X, 3))
    s = rng.uniform(0.3, 3.0, (H, 3))
    s /= s.mean(axis=0)
    T = 100 * rng.dirichlet(np.ones(X), H).T
    n = rng.poisson(lam[:, None, :] * s[None] * T[..., None])
    f = qr.Fit(
        T.sum(),
        n.sum(axis=1)[None, :, None],
        T.sum(axis=1)[None, :, None],
        np.ones(1, dtype=np.int64),
        (1, 2),
        np.zeros((0, 6), dtype=np.int64),
        np.zeros((1, 1), dtype=np.int64),
        100.0 * np.arange(H + 1),
        T[None, :, None],
        n.sum(axis=0),
    )
    return f, s, T


def test_clock_recovers_the_multipliers():
    f, s, _ = synthetic(1)
    got = np.array(f.clock().scale)
    assert np.all(np.abs(got / s - 1) < 4 / np.sqrt(f.arrivals))


def test_clock_matches_arrivals_in_every_slot():
    f, _, T = synthetic(2)
    s = np.array(f.clock().scale)
    lam = f.counts.reshape(-1, 3) / (T @ s)
    assert s * (T.T @ lam) == pytest.approx(f.arrivals, rel=1e-8)
    assert np.diff(f.slots) @ s / f.T == pytest.approx(np.ones(3))


def test_shared_clock_is_one_multiplier():
    f, _, T = synthetic(3)
    s = np.array(f.clock(SHARED).scale)
    assert np.all(s == s[:, :1])
    lam = f.counts.reshape(-1, 3) / (T @ s)
    expected = (s * (T.T @ lam)).sum(axis=1)
    assert expected == pytest.approx(f.arrivals.sum(axis=1), rel=1e-8)
