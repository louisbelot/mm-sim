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


def test_ls_prices_are_the_cost_gradient():
    mm = pm.LSLMSR(0.05, 1)
    rng = np.random.default_rng(8)
    q = rng.uniform(0.5, 40.0, (20, 2))
    h = 1e-6
    for i, e in enumerate(np.eye(2)):
        fd = (mm.cost(q + h * e) - mm.cost(q - h * e)) / (2 * h)
        assert mm.prices(q)[:, i] == pytest.approx(fd, rel=1e-7, abs=1e-8)


def test_ls_is_homogeneous_with_a_bounded_vig():
    v = 0.08
    mm = pm.LSLMSR(v, 1)
    q = np.random.default_rng(9).uniform(0.1, 30.0, (50, 2))
    assert mm.cost(3.7 * q) == pytest.approx(3.7 * mm.cost(q), rel=1e-12)
    assert mm.prices(3.7 * q) == pytest.approx(mm.prices(q), rel=1e-12)
    total = mm.prices(q).sum(axis=1)
    assert np.all(total >= 1) and np.all(total <= 1 + v + 1e-12)
    even = mm.prices(np.array([[5.0, 5.0]])).sum()
    assert even == pytest.approx(1 + v)


@pytest.mark.parametrize("p0", [0.5, 0.2, 0.75])
def test_ls_starts_at_the_prior(p0):
    assert pm.LSLMSR(0.05, 3, p0).price == pytest.approx(np.full(3, p0), rel=1e-9)


def test_ls_loss_is_at_most_v_times_seed():
    v, x = 0.05, 2.0
    r = pm.run(lambda n: pm.LSLMSR(v, n, x=x), 1.0, 400, 200, seed=10)
    assert np.all(r.maker >= -v * x - 1e-9)
    assert r.maker.min() < -0.9 * v * x


def test_ls_deepens_with_volume():
    mm = pm.LSLMSR(0.05, 1, x=1.0)
    up0, _ = mm.quote()
    for _ in range(200):
        mm.buy(np.array([True]))
        mm.buy(np.array([False]))
    up1, _ = mm.quote()
    p = mm.prices()[0, 0]
    assert up1[0] - p < 0.05 * (up0[0] - pm.LSLMSR(0.05, 1, x=1.0).prices()[0, 0])


def test_ls_pnl_is_revenue_minus_payout():
    v, x = 0.05, 1.5
    made = []

    def make(n):
        made.append(pm.LSLMSR(v, n, 0.4, x=x))
        return made[0]

    r = pm.run(make, 0.4, 150, 300, rho=0.7, p0=0.4, seed=11)
    mm = made[0]
    q0 = pm.LSLMSR(v, 1, 0.4, x=x).q[0]
    sold = mm.q - q0
    payout = np.where(r.outcome, sold[:, 0], sold[:, 1])
    revenue = mm.cost(mm.q) - mm.cost(q0[None, :])
    assert r.maker == pytest.approx(revenue - payout, abs=1e-9)


def test_belief_is_bayes_with_the_price_as_prior():
    p = np.array([0.1, 0.5, 0.8])
    rho = 0.7
    for signal, lr in ((True, rho / (1 - rho)), (False, (1 - rho) / rho)):
        odds = p / (1 - p) * lr
        assert pm.belief(p, signal, rho) == pytest.approx(odds / (1 + odds))


@pytest.mark.parametrize(
    "make",
    [lambda n: pm.LMSR(5.0, n), lambda n: pm.FPMM(5.0, n), lambda n: pm.LSLMSR(0.05, n)],
)
def test_useless_signals_never_trade(make):
    r = pm.run(make, 1.0, 50, 100, rho=0.5, seed=12)
    assert r.volume.sum() == 0 and np.all(r.maker == 0)


def test_lmsr_log_score_holds_when_bettors_decline():
    b = 0.8
    made = []
    r = pm.run(lambda n: made.append(pm.LMSR(b, n)) or made[0], 0.6, 200, 500, rho=0.65, seed=13)
    assert 0 < r.volume.mean() < 200
    z = np.where(r.outcome, 1, -1) * made[0].q / b
    assert r.maker == pytest.approx(b * (np.logaddexp(0, -z) + math.log(0.5)), abs=1e-9)


def test_bayes_benchmark_has_the_binomial_brier():
    from scipy.special import expit
    from scipy.stats import binom

    T, rho = 12, 0.6
    r = pm.run(lambda n: pm.LMSR(5.0, n), 1.0, T, 40_000, rho=rho, seed=14)
    j = np.arange(T + 1)
    post = expit((2 * j - T) * math.log(rho / (1 - rho)))
    want = float(np.sum(binom.pmf(j, T, rho) * (1 - post) ** 2))
    err = (r.bayes - r.outcome) ** 2
    assert abs(err.mean() - want) < 4 * err.std() / math.sqrt(len(err))
    assert r.brier(r.bayes) == pytest.approx(err.mean())


def test_lmsr_at_the_flow_likelihood_is_the_glosten_milgrom_belief():
    alpha, rho = 0.3, 0.7
    a = alpha * (2 * rho - 1)
    b = 1 / math.log((1 + a) / (1 - a))
    lm = pm.run(lambda n: pm.LMSR(b, n), alpha, 80, 400, rho=rho, seed=15)
    gm = pm.run(lambda n: pm.Quoter(a, n), alpha, 80, 400, rho=rho, seed=15)
    assert lm.volume.min() == 80 and gm.volume.min() == 80
    assert lm.price == pytest.approx(gm.price, rel=1e-9)


@pytest.mark.parametrize("assumed, trades", [(0.19, True), (0.21, False), (0.3, False)])
def test_quoter_shuts_out_bettors_once_it_assumes_2rho_minus_1(assumed, trades):
    r = pm.run(lambda n: pm.Quoter(assumed, n), 1.0, 30, 200, rho=0.6, seed=16)
    assert (r.volume.sum() > 0) == trades
