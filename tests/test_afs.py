import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import minimize

from mmsim.afs import Block, Power, Profile, continuous, cost, optimal
from mmsim.execution import ObizhaevaWang

SHAPES = [Block(4.0), Power(5.0, 0.5), Power(2.0, -1.0), Profile((2.6, 1.9, 1.6, 1.6, 1.4, 1.2))]
MODELS = ["volume", "price"]


@pytest.mark.parametrize("shape", SHAPES)
def test_shape_calculus(shape):
    for x in (0.0, 0.3, 1.0, 2.7, 5.5, 9.2):
        assert shape.inv(shape.F(x)) == pytest.approx(x, abs=1e-12)
        assert shape.F(x) == pytest.approx(quad(shape.f, 0.0, x, limit=200)[0], rel=1e-7)
        area = quad(lambda u: u * shape.f(u), 0.0, x, limit=200)[0]
        assert shape.G(x) == pytest.approx(area, rel=1e-7, abs=1e-12)


def test_flat_profile_is_a_block():
    p, b = Profile((3.0,) * 4), Block(3.0)
    for x in (0.0, 0.5, 2.0, 3.9, 7.0):
        assert p.F(x) == pytest.approx(b.F(x))
        assert p.G(x) == pytest.approx(b.G(x))
        assert p.inv(3.0 * x) == pytest.approx(b.inv(3.0 * x))


def test_bad_shapes():
    with pytest.raises(ValueError):
        Power(1.0, 1.0)
    with pytest.raises(ValueError):
        Profile((1.0, 0.0))
    with pytest.raises(ValueError):
        optimal(Block(1.0), 10.0, 0, 0.5, "volume")
    with pytest.raises(ValueError):
        optimal(Block(1.0), 10.0, 4, 1.0, "price")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("rho", [0.5, 2.0, 20.0])
def test_block_book_is_obizhaeva_wang(model, rho):
    q, X, T, N = 5.0, 100.0, 1.0, 10
    ow = ObizhaevaWang(1.0 / q, rho)
    xi = optimal(Block(q), X, N, math.exp(-rho * T / N), model)
    assert xi == pytest.approx(ow.trades(X, T, N), rel=1e-10)
    assert cost(Block(q), xi, math.exp(-rho * T / N), model) == pytest.approx(
        ow.cost(xi, np.linspace(0.0, T, N + 1)), rel=1e-10
    )


@pytest.mark.parametrize("model", MODELS)
def test_block_cost_on_any_schedule(model):
    ow = ObizhaevaWang(0.25, 1.3)
    xi = np.array([3.0, 0.5, 2.0, 1.0, 4.0])
    t = np.linspace(0.0, 2.0, 5)
    assert cost(Block(4.0), xi, math.exp(-1.3 * 0.5), model) == pytest.approx(ow.cost(xi, t))


def test_cost_walks_the_book():
    shape = Power(5.0, 0.5)
    a = 0.6
    first = shape.G(shape.inv(3.0))
    e = a * 3.0
    up = shape.inv(e + 2.0)
    assert cost(shape, [3.0, 2.0], a, "volume") == pytest.approx(
        first + shape.G(up) - shape.G(shape.inv(e))
    )
    d = a * shape.inv(3.0)
    up = shape.inv(shape.F(d) + 2.0)
    assert cost(shape, [3.0, 2.0], a, "price") == pytest.approx(first + shape.G(up) - shape.G(d))


@pytest.mark.parametrize("shape", SHAPES[1:])
@pytest.mark.parametrize("model", MODELS)
def test_no_resilience_trades_evenly(shape, model):
    assert optimal(shape, 12.0, 5, 0.0, model) == pytest.approx(np.full(6, 2.0), rel=1e-10)


@pytest.mark.parametrize(
    "shape, model",
    [
        (Power(5.0, 0.5), "volume"),
        (Power(5.0, 0.5), "price"),
        (Power(2.0, -1.0), "volume"),
        (Profile((2.6, 1.9, 1.6, 1.6, 1.4, 1.2)), "volume"),
    ],
)
def test_beats_a_numerical_search(shape, model):
    X, N, a = 30.0, 6, math.exp(-0.5)
    xi = optimal(shape, X, N, a, model)
    assert xi.sum() == pytest.approx(X) and np.all(xi > 0)
    res = minimize(
        lambda x: cost(shape, x, a, model),
        np.full(N + 1, X / (N + 1)),
        method="SLSQP",
        bounds=[(0.0, X)] * (N + 1),
        constraints={"type": "eq", "fun": lambda x: x.sum() - X},
        options={"ftol": 1e-13, "maxiter": 500},
    )
    best = cost(shape, xi, a, model)
    assert best <= res.fun + 1e-9
    assert res.x == pytest.approx(xi, abs=2e-3 * X)


@pytest.mark.parametrize("model", MODELS)
def test_perturbations_cost_more(model):
    shape, X, N, a = Power(5.0, 0.5), 30.0, 8, math.exp(-0.3)
    xi = optimal(shape, X, N, a, model)
    best = cost(shape, xi, a, model)
    rng = np.random.default_rng(1)
    for _ in range(30):
        d = rng.normal(scale=0.3, size=N + 1)
        assert cost(shape, xi + d - d.mean(), a, model) > best


def test_volume_and_price_part_on_a_curved_book():
    shape, a = Power(5.0, 0.5), math.exp(-0.5)
    v, p = optimal(shape, 30.0, 6, a, "volume"), optimal(shape, 30.0, 6, a, "price")
    assert abs(v[0] - p[0]) > 0.1
    assert cost(shape, v, a, "volume") < cost(shape, p, a, "volume")
    assert cost(shape, p, a, "price") < cost(shape, v, a, "price")


@pytest.mark.parametrize("shape", SHAPES)
def test_continuous_limit(shape):
    X, rT, N = 30.0, 3.0, 20_000
    x0, flow, xT = continuous(shape, X, rT)
    xi = optimal(shape, X, N, math.exp(-rT / N), "volume")
    assert xi[0] == pytest.approx(x0, rel=1e-3)
    assert xi[1:-1].sum() == pytest.approx(flow, rel=1e-3)
    assert xi[-1] == pytest.approx(xT, rel=1e-3)


def test_continuous_block_is_ow():
    x0, flow, xT = continuous(Block(5.0), 100.0, 4.0)
    assert x0 == pytest.approx(100.0 / 6.0) and xT == pytest.approx(x0)
    assert flow == pytest.approx(4.0 * x0)
