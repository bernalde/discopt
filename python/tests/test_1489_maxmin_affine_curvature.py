"""Regression tests for #1489.

``max`` / ``min`` of >= 2 affine arguments must not be classified AFFINE: the
pointwise max of affine functions is CONVEX and the min is CONCAVE. The AFFINE
verdict let nonconvex rows (``max(L) >= c``, ``min(L) == 0``) pass the convexity
test, so ``Model.solve()`` took the single-NLP convex fast path and certified a
local optimum as global.
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest
from discopt._relax.convexity.lattice import Curvature
from discopt._relax.convexity.rules import classify_expr, classify_model

pytestmark = pytest.mark.smoke


def _xy():
    m = dm.Model("cls")
    x = m.continuous("x", lb=-3, ub=3)
    y = m.continuous("y", lb=-3, ub=3)
    return m, x, y


def test_max_of_affine_is_convex():
    m, x, y = _xy()
    assert classify_expr(dm.maximum(x, -x), m) == Curvature.CONVEX
    assert classify_expr(dm.maximum(x, y), m) == Curvature.CONVEX
    assert classify_expr(dm.maximum(x + y, 2 - x, y), m) == Curvature.CONVEX


def test_min_of_affine_is_concave():
    m, x, y = _xy()
    assert classify_expr(dm.minimum(x, -x), m) == Curvature.CONCAVE
    assert classify_expr(dm.minimum(x, y), m) == Curvature.CONCAVE


def test_composites_on_min_of_affine_not_convex():
    """The fuzz-found composites built on the wrong verdict."""
    m, x, y = _xy()
    assert classify_expr(dm.abs(dm.minimum(x, y)), m) != Curvature.CONVEX
    assert classify_expr(dm.minimum(x, y) ** 4, m) != Curvature.CONVEX


def test_nonconvex_rows_not_convex_model():
    m, x, y = _xy()
    m.subject_to(dm.maximum(x, -x) >= 2)
    m.minimize(x)
    is_convex, _mask = classify_model(m)
    assert not is_convex

    m2, x2, y2 = _xy()
    m2.subject_to(dm.minimum(x2, y2) == 0)
    m2.maximize(2 * x2 + y2)
    is_convex2, _mask2 = classify_model(m2)
    assert not is_convex2


def test_convex_uses_still_convex():
    """Soundness fix must not lose the legitimate convex verdicts."""
    m, x, y = _xy()
    m.subject_to(dm.maximum(x, -x) <= 2)
    m.subject_to(dm.minimum(x, y) >= -1)
    m.minimize(dm.maximum(x, y))
    is_convex, _mask = classify_model(m)
    assert is_convex


def _build(name):
    m = dm.Model(name)
    if name == "max_x_negx_ge2_min_x":
        x = m.continuous("x", lb=-3, ub=3)
        m.subject_to(dm.maximum(x, -x) >= 2)
        m.minimize(x)
        return m, -3.0, True
    if name == "min_xy_eq0_max_2xpy":
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        m.subject_to(dm.minimum(x, y) == 0)
        m.maximize(2 * x + y)
        return m, 20.0, False
    if name == "max_x_2mx_ge3_max_x":
        x = m.continuous("x", lb=-3, ub=3)
        m.subject_to(dm.maximum(x, 2 - x) >= 3)
        m.maximize(x)
        return m, 3.0, False
    if name == "max_xy_ge2_min_2xpy":
        x = m.continuous("x", lb=-3, ub=3)
        y = m.continuous("y", lb=-3, ub=3)
        m.subject_to(dm.maximum(x, y) >= 2)
        m.minimize(2 * x + y)
        return m, -4.0, True
    assert name == "max_s_negs_ge2"
    x = m.continuous("x", lb=-3, ub=3)
    y = m.continuous("y", lb=-3, ub=3)
    m.subject_to(dm.maximum(x + y, -x - y) >= 2)
    m.minimize(x + 0.5 * y)
    return m, -4.5, True


@pytest.mark.parametrize(
    "name",
    [
        "max_x_negx_ge2_min_x",
        "min_xy_eq0_max_2xpy",
        "max_x_2mx_ge3_max_x",
        "max_xy_ge2_min_2xpy",
        "max_s_negs_ge2",
    ],
)
def test_issue_repros_not_falsely_certified(name):
    m, true_opt, sense_min = _build(name)
    r = m.solve(time_limit=30)
    info = (r.status, r.objective, r.bound, r.gap_certified)
    if r.status == "optimal" or r.gap_certified:
        assert r.objective == pytest.approx(true_opt, abs=1e-4), info
    if r.bound is not None:
        # Certificate invariant: the dual bound never crosses the optimum.
        if sense_min:
            assert r.bound <= true_opt + 1e-4, info
        else:
            assert r.bound >= true_opt - 1e-4, info
