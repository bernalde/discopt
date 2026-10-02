"""#1551: no certificate finer than the objective's own float resolution.

``min 6y^2 + By + C`` with ``y`` integer in ``[c, c+3]``, ``B = -(12c+12)`` and
``C = 6c^2+12c`` is the shifted quadratic of #1543 written out in expanded form.
At ``c = 2345678.9`` each term is ~3.3e13 (ulp ~4e-3), so float64 cannot
evaluate the objective to better than ~1e-2 anywhere in the box. The OA
auto-route returned ``objective = bound = -5.9375`` as ``gap_certified=True``
while the exact optimum of the function as written is ``-5.93832`` -- a lower
bound 8.2e-4 above the optimum, past the 5.9e-4 tolerance it was certified at.

The guard measures the objective's evaluation error at the incumbent as the width
of an outward-rounded interval enclosure, and withdraws a certificate (and its
bound) whose pair no longer closes the requested gap once widened by that error.
"""

from __future__ import annotations

import math
from fractions import Fraction as F

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling.core import SolveResult
from discopt.solver import (
    _objective_evaluation_error,
    _withhold_unresolved_objective_certificate,
    solve_model,
)

BAD_C = 2345678.9


def _expanded(c: float, sense: str = "min"):
    B = -(12 * c + 12)
    C = 6 * c * c + 12 * c
    m = dm.Model("exp")
    y = m.integer("y", lb=c, ub=c + 3)
    f = 6 * y * y + B * y + C
    if sense == "min":
        m.minimize(f)
    else:
        m.maximize(-f)
    opt = min(F(6) * k * k + F(B) * k + F(C) for k in range(math.ceil(c), math.floor(c + 3) + 1))
    return m, y, opt


@pytest.mark.parametrize("sense", ["min", "max"])
def test_unresolvable_objective_is_not_certified(sense):
    m, _, opt = _expanded(BAD_C, sense)
    r = m.solve(time_limit=30)
    assert r.x is not None and r.objective is not None
    assert not r.gap_certified
    assert r.status != "optimal"
    assert r.bound is None and not r.bound_valid
    stats = r.solver_stats or {}
    assert stats.get("certificate/objective_unresolved") == 1.0
    # The measured error is what made the certificate unsupportable.
    assert stats["certificate/objective_eval_error"] > 1e-4 * abs(float(opt))


def test_direct_solve_model_call_is_guarded():
    """The ``solve_model`` wrapper applies the guard without ``Model.solve``."""
    m, _, _ = _expanded(BAD_C)
    r = solve_model(m, time_limit=30)
    assert not r.gap_certified
    assert r.bound is None


def test_enclosure_contains_the_exact_value():
    """The measured error is a rigorous enclosure, checked in exact arithmetic."""
    m, y, _ = _expanded(BAD_C)
    from discopt._relax.convexity.interval import Interval
    from discopt._relax.convexity.interval_eval import evaluate_interval

    checked = 0
    for k in range(math.ceil(BAD_C), math.floor(BAD_C + 3) + 1):
        B = -(12 * BAD_C + 12)
        C = 6 * BAD_C * BAD_C + 12 * BAD_C
        exact = F(6) * k * k + F(B) * k + F(C)
        enc = evaluate_interval(m._objective.expression, m, {y: Interval.point(float(k))})
        assert F(float(enc.lo)) <= exact <= F(float(enc.hi))
        err = _objective_evaluation_error(m, {"y": np.array(float(k))})
        assert err == pytest.approx(float(enc.hi - enc.lo))
        checked += 1
    assert checked == 3


def test_well_scaled_model_keeps_its_certificate():
    """The same function at a small offset resolves to the ulp and stays certified."""
    m, _, opt = _expanded(0.9)
    r = m.solve(time_limit=30)
    assert r.status == "optimal" and r.gap_certified
    assert r.bound is not None and F(r.bound) <= opt + F(1e-6)
    stats = r.solver_stats or {}
    assert stats["certificate/objective_eval_error"] < 1e-12
    assert "certificate/objective_unresolved" not in stats


def _certified(obj: float, bound: float, y: float) -> SolveResult:
    return SolveResult(
        status="optimal",
        objective=obj,
        bound=bound,
        gap=0.0,
        x={"y": np.array(y)},
        gap_certified=True,
    )


def test_guard_unit():
    m, _, _ = _expanded(BAD_C)
    y0 = float(math.ceil(BAD_C) + 1)

    bad = _certified(-5.9375, -5.9375, y0)
    _withhold_unresolved_objective_certificate(bad, m, 1e-4, 1e-6)
    assert not bad.gap_certified and bad.status == "feasible" and bad.bound is None

    # A loose caller tolerance absorbs the same error: left alone.
    loose = _certified(-5.9375, -5.9375, y0)
    _withhold_unresolved_objective_certificate(loose, m, 1e-1, 1e-1)
    assert loose.gap_certified and loose.bound == -5.9375

    # Uncertified results are never touched (downgrade-only).
    unc = _certified(-5.9375, -5.9375, y0)
    unc.gap_certified = False
    unc.status = "feasible"
    _withhold_unresolved_objective_certificate(unc, m, 1e-4, 1e-6)
    assert unc.bound == -5.9375


def test_unmeasurable_point_is_not_measured():
    m, _, _ = _expanded(BAD_C)
    assert _objective_evaluation_error(m, {}) is None
    assert _objective_evaluation_error(dm.Model("empty"), {}) is None
