import math

import numpy as np
import pytest

from mmsim import prediction as pm


def test_lmsr_loss_is_log_score_of_the_final_price():
    b, p0 = 7.0, 0.3
    r = pm.run(lambda n: pm.LMSR(b, n, p0), 0.3, 60, 500, p0=p0, seed=1)
    pT = np.where(r.outcome, r.price, 1 - r.price)
    start = np.where(r.outcome, p0, 1 - p0)
    assert r.maker == pytest.approx(-b * np.log(pT / start), abs=1e-9)
    assert np.all(-r.maker <= b * np.log(1 / start) + 1e-12)


def test_fpmm_loss_follows_the_reserve_curve():
    F, p0 = 5.0, 0.6
    r = pm.run(lambda n: pm.FPMM(F, n, p0), 0.3, 60, 500, p0=p0, seed=2)
    p = r.price
    yes = F * (np.sqrt((1 - p) / p) - math.sqrt((1 - p0) / p0))
    no = F * (np.sqrt(p / (1 - p)) - math.sqrt(p0 / (1 - p0)))
    assert r.maker == pytest.approx(np.where(r.outcome, yes, no), abs=1e-9)


def test_all_informed_flow_drives_loss_to_the_bound():
    lmsr = pm.run(lambda n: pm.LMSR(4.0, n), 1.0, 400, 50)
    assert lmsr.maker == pytest.approx(-4.0 * math.log(2), abs=1e-6)
    fpmm = pm.run(lambda n: pm.FPMM(4.0, n), 1.0, 4000, 50)
    assert np.all(fpmm.maker > -4.0) and np.all(fpmm.maker < -3.8)


def test_same_depth_at_one_half_when_b_equals_F():
    L = 1000.0
    for mm in (pm.LMSR(L, 1), pm.FPMM(L, 1)):
        mm.buy(np.array([True]))
        assert mm.price[0] - 0.5 == pytest.approx(1 / (4 * L), rel=1e-3)


@pytest.mark.parametrize("p", [0.1, 0.3, 0.8])
def test_fpmm_is_shallower_than_lmsr_away_from_one_half(p):
    L = 1e4
    lmsr, fpmm = pm.LMSR(L, 1, p), pm.FPMM(L, 1, p)
    for mm in (lmsr, fpmm):
        mm.buy(np.array([True]))
    ratio = (fpmm.price[0] - p) / (lmsr.price[0] - p)
    assert ratio == pytest.approx(2 * math.sqrt(p * (1 - p)), rel=1e-3)


def test_complete_set_costs_one_or_more():
    lmsr = pm.LMSR(3.0, 4, 0.2)
    fpmm = pm.FPMM(3.0, 4, 0.2)
    for mm in (lmsr, fpmm):
        a = mm.buy(np.array([True, False, True, False]))
        b = mm.buy(np.array([False, True, False, True]))
        if mm is lmsr:
            assert a + b == pytest.approx(np.ones(4))
            assert mm.price == pytest.approx(np.full(4, 0.2))
        else:
            assert np.all(a + b >= 1)


def test_quoter_belief_is_the_bayes_posterior():
    alpha, p0 = 0.25, 0.4
    lr = (alpha + (1 - alpha) / 2) / ((1 - alpha) / 2)
    rng = np.random.default_rng(3)
    yes = rng.random((30, 8)) < 0.6
    gm = pm.Quoter(alpha, 8, p0)
    for row in yes:
        gm.buy(row)
    k = yes.sum(axis=0) - (~yes).sum(axis=0)
    odds = p0 / (1 - p0) * lr**k
    assert gm.price == pytest.approx(odds / (1 + odds), rel=1e-10)


def test_quoter_breaks_even_and_noise_pays_the_informed():
    r = pm.run(lambda n: pm.Quoter(0.2, n), 0.2, 100, 20_000, seed=4)
    se = r.maker.std() / math.sqrt(len(r.maker))
    assert abs(r.maker.mean()) < 4 * se
    assert r.maker + r.informed + r.noise == pytest.approx(np.zeros(20_000), abs=1e-9)
    assert r.informed.mean() > 0 > r.noise.mean()


def test_prices_converge_to_the_outcome():
    for make in (lambda n: pm.Quoter(0.3, n), lambda n: pm.LMSR(5.0, n), lambda n: pm.FPMM(5.0, n)):
        r = pm.run(make, 0.3, 300, 400, seed=5)
        assert r.error[-1] < 0.05 < r.error[0]
        assert np.all((r.price > 0) & (r.price < 1))
