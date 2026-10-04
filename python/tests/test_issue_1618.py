"""Regression tests for #1618 -- wrong or missing statuses, duals and diagnostics.

One test per fixed item; each fails on the pre-fix tree. Item IDs are the issue's.
"""

from __future__ import annotations

import logging

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling.core import SolveResult
from discopt.solvers import SolveStatus

# --------------------------------------------------------------------------- A-15


def test_a15_lp_simplex_reports_infeasible_not_error():
    """An LP with ``x0 + x1 <= 1`` and ``x0 + x1 >= 3`` is infeasible; the Rust
    simplex answers ``numerical`` on it and ``solve_lp`` used to return ERROR."""
    from discopt.solvers.lp_simplex import solve_lp

    r = solve_lp(
        np.zeros(2),
        A_ub=np.array([[1.0, 1.0], [-1.0, -1.0]]),
        b_ub=np.array([1.0, -3.0]),
        bounds=[(0, None)] * 2,
    )
    assert r.status == SolveStatus.INFEASIBLE

    r = solve_lp(
        np.zeros(2),
        A_eq=np.array([[1.0, 1.0], [1.0, 1.0]]),
        b_eq=np.array([1.0, 3.0]),
        bounds=[(0, None)] * 2,
    )
    assert r.status == SolveStatus.INFEASIBLE


def test_a15_numerical_on_feasible_lp_stays_error(monkeypatch):
    """Soundness: a ``numerical`` verdict on a FEASIBLE LP must not be upgraded to
    INFEASIBLE -- only a phase-1 proof licenses that."""
    import discopt._rust as rust
    from discopt.solvers.lp_simplex import solve_lp

    calls = []

    def fake(*args, **kwargs):
        calls.append(1)
        return ("numerical", None, None, 0, None, None, None, None)

    monkeypatch.setattr(rust, "solve_lp_warm_csc_py", fake)
    r = solve_lp(
        np.array([1.0, 1.0]),
        A_ub=np.array([[1.0, 1.0]]),
        b_ub=np.array([5.0]),
        bounds=[(0, None)] * 2,
    )
    assert calls, "the patched simplex was never called"
    assert r.status == SolveStatus.ERROR


# -------------------------------------------------------------------------- B-05b


def test_b05b_acceptable_level_is_local_limit_not_local_optimal():
    """POUNCE code 1 (Solved_To_Acceptable_Level) met only the acceptable dual
    tolerance (1e10 by default); it is not an established stationary point."""
    t0 = np.array([10.0, 12.0, 20.0])
    cap = np.array([1000.0, 1500.0, 800.0])
    D, VOT = 2500.0, 0.25
    m = dm.Model("toll")
    tau = m.continuous("tau", lb=0, ub=10)
    x = [m.continuous(f"x{a}", lb=0, ub=D) for a in range(3)]
    z = [m.continuous(f"z{a}", lb=0, ub=100) for a in range(3)]
    pi = m.continuous("pi", lb=0, ub=100)
    for a in range(3):
        m.subject_to(
            t0[a] * (1 + 0.15 * (x[a] / cap[a]) ** 4) + (tau / VOT if a == 1 else 0) - pi - z[a]
            == 0
        )
        m.subject_to(x[a] * z[a] == 0, name=f"pair{a}")
    m.subject_to(sum(x) == D)
    m.maximize(tau * x[1])
    r = m.solve(
        solver="pounce",
        initial_solution={tau: 1.0, pi: 15.0, **{v: D / 3 for v in x}, **{v: 1.0 for v in z}},
    )
    upstream = r.solve_report["solution"]["status_upstream"]
    if "Acceptable" not in str(upstream):
        pytest.skip(f"witness no longer stops at the acceptable level ({upstream!r})")
    assert r.status == "local_limit"


# --------------------------------------------------------------------------- D-37


def _res(obj, bound, gap_certified, gap):
    return SolveResult(
        status="optimal" if gap_certified else "feasible",
        objective=obj,
        bound=bound,
        gap=gap,
        gap_certified=gap_certified,
    )


def test_d37_merge_within_certified_gap_does_not_warn(caplog):
    from discopt.solver import _merge_route_and_fallback

    fb = _res(5822.936290001, 5822.936290001, True, 0.0)
    route = _res(5822.936290000, None, False, None)
    with caplog.at_level(logging.DEBUG, logger="discopt.solver"):
        out = _merge_route_and_fallback(route, fb, is_maximize=False)
    assert out.objective == fb.objective  # the certified result is still kept
    warn = [r for r in caplog.records if r.levelno >= logging.WARNING and "#1059" in r.message]
    assert warn == []
    assert any("within the certified gap" in r.message for r in caplog.records)


def test_d37_merge_beyond_certified_bound_still_warns(caplog):
    from discopt.solver import _merge_route_and_fallback

    fb = _res(10.0, 10.0, True, 0.0)
    route = _res(9.0, None, False, None)
    with caplog.at_level(logging.WARNING, logger="discopt.solver"):
        out = _merge_route_and_fallback(route, fb, is_maximize=False)
    assert out.objective == 10.0
    assert any(
        "should be impossible" in r.message and r.levelno == logging.WARNING for r in caplog.records
    )


# -------------------------------------------------------------------------- B-07b


@pytest.mark.parametrize("sense", ["min", "max"])
@pytest.mark.parametrize("method", ["exact", "fd"])
def test_b07b_predict_objective_matches_finite_difference(sense, method):
    from discopt.solvers.sipopt import pounce_sensitivity

    m = dm.Model("s")
    p = m.parameter("p", 3.0)
    x = m.continuous("x", lb=-10, ub=10)
    y = m.continuous("y", lb=-10, ub=10)
    m.subject_to(x + y == p)
    f = (x - 1) ** 2 + (y - p) ** 2 + p * x
    if sense == "min":
        m.minimize(f)
    else:
        m.maximize(-f)
    s = pounce_sensitivity(m, [p], method=method)

    def fstar(pv):  # internal (minimization) form, the convention of s.objective
        p.value = pv
        try:
            r = m.solve(solver="pounce")
        finally:
            p.value = 3.0
        return r.objective if sense == "min" else -r.objective

    h = 1e-4
    fd = (fstar(3.0 + h) - fstar(3.0 - h)) / (2 * h)
    assert s.dobjective_dp is not None
    assert abs(s.objective - fstar(3.0)) < 1e-6
    assert abs(float(s.dobjective_dp[0]) - fd) < 1e-4
    # first-order prediction: the old predict_objective returned s.objective unchanged
    assert abs(s.predict_objective([3.1]) - fstar(3.1)) < 1e-2
    assert abs(s.predict_objective([3.1]) - s.objective) > 1e-3


# --------------------------------------------------------------------------- A-20


def test_a20_unbounded_inside_default_box_records_cause():
    m = dm.Model("a20")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y")  # default box, not declared free
    m.minimize(x**2 - y)
    with pytest.warns(RuntimeWarning):
        r = m.solve()
    if r.status != "error":
        pytest.skip(f"witness no longer takes the huge-bound refusal ({r.status})")
    assert r.error
    assert "default box" in r.error


# --------------------------------------------------------------------------- A-09


def test_a09_pounce_lp_no_verified_answer_names_engine_reason():
    RON = [93, 70, 98, 92, 96]
    RVP = [52.0, 11, 3.5, 7, 4.6]
    S = [10.0, 5, 1, 20, 6]
    SG = [0.584, 0.664, 0.81, 0.745, 0.7]
    cost = [45, 70, 98, 92, 105]
    m = dm.Model("blend")
    x = {(i, j): m.continuous(f"x{i}{j}", lb=0, ub=np.inf) for i in range(5) for j in range(2)}
    for j, rm in enumerate([91, 96]):
        m.subject_to(sum((RON[i] - rm) * x[i, j] for i in range(5)) >= 0)
        m.subject_to(sum((RVP[i] ** 1.25 - 9**1.25) * x[i, j] for i in range(5)) <= 0)
        m.subject_to(sum(SG[i] * (S[i] - 10) * x[i, j] for i in range(5)) <= 0)
    m.maximize(sum(([110, 122][j] - cost[i]) * x[i, j] for i in range(5) for j in range(2)))
    r = m.solve(solver="pounce")
    if r.status != "error":
        pytest.skip(f"witness now certifies ({r.status})")
    assert "dual_infeasible" in r.error
    assert "not certified" in r.error
