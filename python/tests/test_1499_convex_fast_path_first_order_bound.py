"""Regression tests for #1499 (residue of #853): the convex fast path certified a
point that an exhibited feasible point had already beaten.

``min 1/(x+1)`` on ``[0, 1e6]`` stalls at ``x ~ 11258`` where the gradient has all
but vanished. The #853 witness search found the vertex ``x = 1e6`` beating it by
8.8e-5 -- inside the 1e-4 tolerance -- so the certificate stood and the NLP's
``8.88e-5`` was published as the DUAL BOUND against a true optimum of ``1e-6``.
``solver="amp"`` skipped the gate entirely and also certified ``1000/(x+1)``
(2.4e-3 vs 1e-3) and ``x**-0.5`` on ``[1, 1e8]`` (2.7e-3 vs 1e-4).

The fix: any real improvement makes the certificate rest on a rigorous
first-order (tangent) box bound of the convex Lagrangian, published in place of
``f(x)``, with the better point adopted -- on every route through the fast path.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest

_ROUTES = [None, "bb", "amp"]


def _check(r, true_obj, maximize):
    """Sound certificate: bound never crosses the optimum; certified => correct."""
    assert r.bound is None or np.isfinite(r.bound)
    if r.bound is not None:
        if maximize:
            assert r.bound >= true_obj - 1e-9 * (1 + abs(true_obj)), (r.bound, true_obj)
        else:
            assert r.bound <= true_obj + 1e-9 * (1 + abs(true_obj)), (r.bound, true_obj)
    if r.gap_certified:
        assert r.status == "optimal"
        assert r.objective == pytest.approx(true_obj, rel=1e-4, abs=1e-6), (r.objective, true_obj)


@pytest.mark.parametrize("solver", _ROUTES)
@pytest.mark.parametrize("k", [1.0, 1000.0])
@pytest.mark.parametrize("maximize", [False, True])
def test_shifted_reciprocal_on_wide_box(solver, k, maximize):
    m = dm.Model("recip")
    x = m.continuous("x", lb=0, ub=1e6)
    if maximize:
        m.maximize(-k / (x + 1))
        true_obj = -k / (1e6 + 1)
    else:
        m.minimize(k / (x + 1))
        true_obj = k / (1e6 + 1)
    r = m.solve(time_limit=15, solver=solver)
    _check(r, true_obj, maximize)
    # These are easy convex problems: the fixed path certifies the true optimum.
    assert r.gap_certified, (solver, r.status, r.objective, r.bound)


@pytest.mark.parametrize("solver", _ROUTES)
def test_inverse_sqrt_on_wide_box(solver):
    m = dm.Model("pw")
    x = m.continuous("x", lb=1, ub=1e8)
    m.minimize(x**-0.5)
    r = m.solve(time_limit=15, solver=solver)
    _check(r, 1e-4, maximize=False)


@pytest.mark.parametrize("solver", _ROUTES)
def test_constrained_reciprocal(solver):
    """The rigorous bound also holds with a (linear) row in play."""
    m = dm.Model("recip_con")
    x = m.continuous("x", lb=0, ub=1e6)
    y = m.continuous("y", lb=0, ub=1e6)
    m.subject_to(x + y <= 1e6)
    m.minimize(1 / (x + 1) + 1 / (y + 1))
    r = m.solve(time_limit=15, solver=solver)
    # optimum at x = y = 5e5: 2 / (5e5 + 1)
    _check(r, 2.0 / (5e5 + 1), maximize=False)


def test_genuine_optimum_keeps_its_certificate():
    """A true interior optimum has no better witness and stays certified exactly."""
    m = dm.Model("quad")
    x = m.continuous("x", lb=-10, ub=10)
    m.minimize((x - 1) ** 2 + 3)
    r = m.solve(time_limit=15)
    assert r.gap_certified and r.status == "optimal"
    assert r.objective == pytest.approx(3.0, abs=1e-8)
    assert r.bound == pytest.approx(3.0, abs=1e-6)


def test_tangent_box_bound_closed_form():
    from discopt.solver import _tangent_box_bound

    lo, hi = np.array([0.0, -1.0]), np.array([2.0, np.inf])
    s = np.array([1.0, 0.0])
    # grad (+1, 0): min at x0 = 0 -> value - 1; zero component ignores the inf bound.
    assert _tangent_box_bound(5.0, np.array([1.0, 0.0]), s, lo, hi) == 4.0
    # a nonzero component pointing at an infinite bound gives no bound
    assert _tangent_box_bound(5.0, np.array([1.0, -1e-3]), s, lo, hi) == -np.inf
    # non-finite slope gives no bound (#1491)
    assert _tangent_box_bound(5.0, np.array([np.inf, 0.0]), s, lo, hi) == -np.inf
