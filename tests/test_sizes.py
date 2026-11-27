import math
import random

import numpy as np
import pytest
from scipy.special import zeta

from mmsim.sizes import Sizes, fit_geometric, fit_zeta, units, vuong, zeta_alpha


def sample(s, n, seed):
    rng = random.Random(seed)
    return np.array([s.draw(rng) for _ in range(n)])


def test_rejects_bad_pmf():
    for bad in ((), (0.5, 0.4), (1.2, -0.2)):
        with pytest.raises(ValueError):
            Sizes(bad)


def test_unit_always_one():
    assert set(sample(Sizes.unit(), 100, 1)) == {1}
    assert Sizes.unit().mean == 1


def test_long_zeta_head_is_riemann_zeta():
    s = Sizes.zeta(2.5, 1_000_000)
    assert s.pmf[0] == pytest.approx(1 / zeta(2.5), rel=1e-8)
    assert s.mean == pytest.approx(zeta(1.5) / zeta(2.5), rel=1e-3)
    assert zeta_alpha(s) == pytest.approx(2.5)


def test_draw_frequencies():
    s = Sizes.zeta(2.0, 8)
    n = 200_000
    counts = np.bincount(sample(s, n, 2), minlength=9)[1:]
    p = np.array(s.pmf)
    assert np.all(np.abs(counts - n * p) < 4 * np.sqrt(n * p * (1 - p)))


def test_itch_mean():
    s = Sizes.itch()
    assert s.cap == 20
    assert 1.4 < s.mean < 1.6


def test_units_round_and_clip():
    got = units(np.array([10, 99, 151, 249, 10_000]), 100.0, 20)
    assert got.tolist() == [1, 1, 2, 2, 20]


@pytest.mark.parametrize("alpha", [1.5, 2.7, 3.5])
def test_zeta_mle_recovers_alpha(alpha):
    cap, n = 50, 40_000
    s = Sizes.zeta(alpha, cap)
    log_k = np.log(np.arange(1, cap + 1))
    var = np.array(s.pmf) @ log_k**2 - (np.array(s.pmf) @ log_k) ** 2
    got = zeta_alpha(fit_zeta(sample(s, n, int(10 * alpha)), cap))
    assert abs(got - alpha) < 4 / math.sqrt(n * var)


def test_geometric_mle_recovers_q():
    cap, n, q = 50, 40_000, 0.4
    s = Sizes.geometric(q, cap)
    got = fit_geometric(sample(s, n, 5), cap)
    se = (1 - q) * math.sqrt(q) / math.sqrt(n)
    assert abs(got.pmf[1] / got.pmf[0] - q) < 4 * se
    assert got.mean == pytest.approx(sample(s, n, 5).mean(), rel=1e-9)


def test_vuong_picks_the_true_law():
    cap = 30
    for truth, sign in ((Sizes.zeta(2.5, cap), 1), (Sizes.geometric(0.35, cap), -1)):
        k = sample(truth, 5000, 9)
        _, z = vuong(fit_zeta(k, cap), fit_geometric(k, cap), k)
        assert sign * z > 3
