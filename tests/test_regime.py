import numpy as np
import pytest
from scipy.stats import norm

from mmsim import regime as g

SMALL = g.World(T=0.3, Q=15, phi=0.01, eta=0.05)


def test_world_rejects_bad_parameters():
    with pytest.raises(ValueError):
        g.World(A=250.0)
    with pytest.raises(ValueError):
        g.World(sigma=(0.0, 2.0))
    with pytest.raises(ValueError):
        g.World(rate=(-1.0, 1.0))
    with pytest.raises(ValueError):
        g.World(pi0=1.5)


def test_update_is_bayes_then_one_markov_step():
    w = g.World(b=(-3.0, 4.0), sigma=(1.0, 3.0), rate=(5.0, 1.0))
    rng = np.random.default_rng(0)
    pi, dz = rng.random(50), 0.3 * rng.standard_normal(50)
    l0 = norm.pdf(dz, w.b[0] * w.dt, w.sigma[0] * np.sqrt(w.dt))
    l1 = norm.pdf(dz, w.b[1] * w.dt, w.sigma[1] * np.sqrt(w.dt))
    post = np.stack([(1 - pi) * l0, pi * l1]) / ((1 - pi) * l0 + pi * l1)
    assert np.allclose(g.update(w, pi, dz), (w.P.T @ post)[1], atol=1e-14)


def test_certain_beliefs_stay_certain_without_switching():
    w = g.World(rate=(0.0, 0.0))
    dz = np.linspace(-1, 1, 7)
    assert np.all(g.update(w, 0.0, dz) == 0.0)
    assert np.all(g.update(w, 1.0, dz) == 1.0)
