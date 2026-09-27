"""#1496: ``nonlinear_to_pwl`` must publish incumbents inside the declared box.

``verify_point`` accepts a point up to tolerance outside the bounds. The polished
(or mapped) point used to be kept raw, so its objective could beat every in-box
point: ``min (x-0.3)**2`` on ``[1e4, 1e4+10]`` returned x = 9999.9999 with an
objective below the certified bound, and ``max exp(x)`` on ``[0, 18]`` evaluated
past e^18 and tripped the soundness tripwire on a perfectly valid bound.
The candidate is now clipped into the box (integers rounded) and re-verified.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest


def _check_result(r, lb, ub, maximize):
    xv = float(np.asarray(r.x["x"]).ravel()[0])
    assert lb <= xv <= ub, f"incumbent x = {xv!r} outside [{lb}, {ub}]"
    assert r.bound is not None and r.objective is not None
    slack = 1e-9 * max(1.0, abs(r.objective))
    if maximize:
        assert r.objective <= r.bound + slack, (r.objective, r.bound)
    else:
        assert r.objective >= r.bound - slack, (r.objective, r.bound)
    return 2


def test_issue_repro_max_exp_does_not_trip():
    m = dm.Model("a")
    x = m.continuous("x", lb=0, ub=18)
    m.maximize(dm.exp(x))
    r = dm.nonlinear_to_pwl(m).solve(time_limit=30)
    assert r.status == "optimal" and r.gap_certified
    assert _check_result(r, 0.0, 18.0, True) == 2
    assert r.objective <= np.exp(18.0) * (1 + 1e-12)


def test_issue_repro_min_square_objective_not_below_bound():
    m = dm.Model("b")
    x = m.continuous("x", lb=1e4, ub=1e4 + 10)
    m.minimize((x - 0.3) ** 2)
    r = dm.nonlinear_to_pwl(m).solve(time_limit=30)
    assert r.status == "optimal" and r.gap_certified
    assert _check_result(r, 1e4, 1e4 + 10, False) == 2
    assert r.objective >= (1e4 - 0.3) ** 2 * (1 - 1e-12)


@pytest.mark.parametrize("ub", [5.0, 12.0, 18.0])
def test_max_exp_panel(ub):
    m = dm.Model("e")
    x = m.continuous("x", lb=0.0, ub=ub)
    m.maximize(dm.exp(x))
    r = dm.nonlinear_to_pwl(m).solve(time_limit=30)
    assert _check_result(r, 0.0, ub, True) == 2


@pytest.mark.parametrize("off", [1e2, 1e3, 1e4])
def test_min_square_at_the_lower_bound_panel(off):
    """Optimum at the lower bound: the polish NLP tends to step just past it."""
    m = dm.Model("s")
    x = m.continuous("x", lb=off, ub=off + 10)
    m.minimize((x - 0.3) ** 2)
    r = dm.nonlinear_to_pwl(m).solve(time_limit=30)
    assert _check_result(r, off, off + 10, False) == 2


# ---------------------------------------------------------------------------
# The issue's secondary ask: among verified candidates, prefer the less violated.
# Choosing purely by objective picked the PWL-mapped point that spent the
# feasibility tolerance on a binding equality to beat the true optimum -- its
# objective landed BELOW the certified bound (9 of 24 equality panels before).
# ---------------------------------------------------------------------------


def _cube(c):
    m = dm.Model("cube")
    x = m.continuous("x", lb=c, ub=c + 10)
    y = m.continuous("y", lb=-1e12, ub=1e12)
    m.subject_to(y == x**3 / (1 + c) ** 2)
    m.minimize(y - 2 * x)
    return m


def _sqrt(c):
    m = dm.Model("sqrt")
    x = m.continuous("x", lb=c, ub=c + 10)
    y = m.continuous("y", lb=-1e12, ub=1e12)
    m.subject_to(y == dm.sqrt(x + 1) * 10)
    m.minimize(-y + 0.01 * x)
    return m


@pytest.mark.parametrize(
    "build, c",
    [(_cube, 10.0), (_cube, 1e2), (_cube, 1e3), (_sqrt, 0.0), (_sqrt, 1e2), (_sqrt, 1e4)],
)
def test_equality_panel_incumbent_not_past_bound_and_least_violating(build, c):
    from discopt.validation.feasibility import max_constraint_violation

    m = build(c)
    r = dm.nonlinear_to_pwl(m).solve(time_limit=30)
    assert r.status == "optimal" and r.gap_certified, r.status
    assert r.objective is not None and r.bound is not None
    # Minimisation: a verified incumbent can never sit below a valid bound.
    assert r.objective >= r.bound - 1e-9 * max(1.0, abs(r.objective)), (r.objective, r.bound)
    xc = np.array([float(np.ravel(r.x["x"])[0]), float(np.ravel(r.x["y"])[0])])
    # The polished point (violation ~1e-11) is available on every one of these;
    # the mapped point it used to lose to sits at 1e-7 .. 3e-5.
    assert max_constraint_violation(m, xc) < 1e-9


def test_prefer_candidate_rule():
    from discopt.modeling._pwl_transform import _prefer_candidate

    x = np.zeros(1)
    clean = (x, 10.0, 1e-11)
    dirty_tiny_gain = (x, 10.0 - 3e-5, 3e-5)  # gain inside rel 1e-4: invisible
    dirty_big_gain = (x, 9.0, 3e-5)  # gain far outside the tolerance
    kw = dict(maximize=False, rel=1e-4, abs_tol=1e-6)
    assert _prefer_candidate(clean, None, **kw)
    assert not _prefer_candidate(dirty_tiny_gain, clean, **kw)
    assert _prefer_candidate(clean, dirty_tiny_gain, **kw)
    assert _prefer_candidate(dirty_big_gain, clean, **kw)
    # Equal violation: objective decides even inside the tolerance.
    assert _prefer_candidate((x, 10.0 - 1e-7, 1e-11), clean, **kw)
    assert not _prefer_candidate((x, 10.0 + 1e-7, 1e-11), clean, **kw)
    # Maximisation mirrors it.
    kw_max = dict(maximize=True, rel=1e-4, abs_tol=1e-6)
    assert not _prefer_candidate((x, 10.0 + 3e-5, 3e-5), clean, **kw_max)
    assert _prefer_candidate((x, 11.0, 3e-5), clean, **kw_max)


def test_max_constraint_violation_measure():
    from discopt.validation.feasibility import max_constraint_violation

    m = dm.Model("v")
    x = m.continuous("x", lb=-10, ub=10)
    y = m.continuous("y", lb=-10, ub=10)
    m.subject_to(x + y == 1)
    m.subject_to(x <= 2)
    m.minimize(x)
    assert max_constraint_violation(m, np.array([0.5, 0.5])) == 0.0
    assert max_constraint_violation(m, np.array([0.5, 0.5 + 1e-3])) == pytest.approx(1e-3)
    assert max_constraint_violation(m, np.array([2.5, -1.5])) == pytest.approx(0.5)
