import itertools

import numpy as np
import pytest

from mmsim import glosten as gm


def normal_grid(mean=100.0, sd=10.0, n=241):
    v = np.linspace(mean - 4 * sd, mean + 4 * sd, n)
    p = np.exp(-0.5 * ((v - mean) / sd) ** 2)
    return v, p / p.sum()


@pytest.mark.parametrize("mu, alpha, eta", [(0.5, 0.2, 0.5), (0.7, 0.4, 0.5), (0.2, 0.6, 0.3)])
def test_two_point_grid_matches_closed_form(mu, alpha, eta):
    v, p = np.array([90.0, 110.0]), np.array([1 - mu, mu])
    bid, ask = gm.quotes(gm.Model(alpha, eta=eta), v, p)
    assert (bid, ask) == pytest.approx(gm.binary_quotes(mu, alpha, 90.0, 110.0, eta), abs=1e-9)


def test_even_prior_spread_is_alpha_times_range():
    for alpha in (0.1, 0.5, 0.9):
        bid, ask = gm.binary_quotes(0.5, alpha, 0.0, 1.0)
        assert ask - bid == pytest.approx(alpha)


def test_belief_depends_only_on_net_order_flow():
    m, v = gm.Model(0.3), np.array([0.0, 1.0])
    r = (1 + m.alpha) / (1 - m.alpha)
    finals = set()
    for sides in set(itertools.permutations([gm.BUY] * 4 + [gm.SELL] * 2)):
        p = np.array([0.6, 0.4])
        for s in sides:
            p = gm.update(m, v, p, s, *gm.quotes(m, v, p))
        finals.add(round(p[1], 12))
    assert finals == {round(0.4 * r**2 / (0.4 * r**2 + 0.6), 12)}


def test_quotes_are_conditional_expectations():
    v, p = normal_grid()
    m = gm.Model(0.3, noise=5.0, eta=0.4)
    bid, ask = gm.quotes(m, v, p)
    for side, px in ((gm.BUY, ask), (gm.SELL, bid)):
        assert gm.update(m, v, p, side, bid, ask) @ v == pytest.approx(px, abs=1e-9)
    assert bid < p @ v < ask


def test_next_mean_is_a_martingale():
    v, p = normal_grid()
    p = p * np.exp(0.05 * (v - 100))
    p /= p.sum()
    m = gm.Model(0.3, noise=5.0, eta=0.4)
    bid, ask = gm.quotes(m, v, p)
    expected = 0.0
    for side in (gm.BUY, gm.NONE, gm.SELL):
        prob = p @ gm.likelihood(m, v, side, bid, ask)
        expected += prob * (gm.update(m, v, p, side, bid, ask) @ v)
    assert expected == pytest.approx(p @ v, abs=1e-10)


def test_noisier_insiders_mean_a_tighter_spread():
    v, p = normal_grid()
    spreads = []
    for noise in (0.01, 5.0, 20.0, 100.0):
        bid, ask = gm.quotes(gm.Model(0.3, noise=noise), v, p)
        spreads.append(ask - bid)
    assert spreads == sorted(spreads, reverse=True)
    assert spreads[-1] < 0.1 * spreads[0]


def test_market_maker_breaks_even_and_insiders_win():
    v, prior = normal_grid(n=81)
    m = gm.Model(0.3, noise=5.0, eta=0.4)
    rng = np.random.default_rng(7)
    mm, insiders = [], []
    for seed in range(600):
        value = rng.choice(v, p=prior)
        path = gm.simulate(m, v, prior, value, 20, seed=seed)
        pnl = path.mm_pnl(value)
        mm.append(pnl.sum())
        insiders.append(-pnl[path.informed].sum())
    mm, insiders = np.array(mm), np.array(insiders)
    assert abs(mm.mean()) < 4 * mm.std() / np.sqrt(len(mm))
    assert insiders.mean() > 8 * insiders.std() / np.sqrt(len(insiders))


def test_prices_find_the_value_and_the_spread_closes():
    v, prior = normal_grid()
    path = gm.simulate(gm.Model(0.5, noise=2.0), v, prior, 112.0, 400, seed=3)
    spread = path.ask - path.bid
    assert abs(path.mean[-1] - 112.0) < 1.0
    assert spread[-1] < 0.1 * spread[0]
