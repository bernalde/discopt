"""Status honesty for ``norm(v, 'inf')`` and ``x**-2 == c`` (#1493).

Two results whose status contradicted what the solve had actually established.

1. ``min norm(v, inf)`` s.t. ``v0 + v1 >= 1`` (convex, optimum 0.5) came back
   ``status="error"`` beside a feasible point ``(0.545, 0.545)`` and
   ``error=None``. The model is certified convex, the single NLP stalls at the
   max-kink and ends with a non-optimal Ipopt code that ``_IPOPT_STATUS_MAP``
   collapses to ``ERROR``; the point then PASSES the false-primal screen and was
   returned under ``error`` anyway. A screened point with no optimality proof is
   ``feasible`` (no bound), and an ``error`` status must say what failed.

2. ``max x`` s.t. ``x**-2 == 4`` on ``[-3, 3]`` (optimum 0.5) came back
   ``unknown`` with no point. Root cause is an UNSOUND envelope: ``_pow_curv``
   declared every even integer power "convex on all of R", including ``t**-2``,
   which has a pole at 0. On ``[-3, 3]`` the builder emitted the secant
   ``w <= 1/9`` and the endpoint tangents as global underestimators, so the root
   LP was empty for a feasible model. Only the missing Farkas ray kept that from
   fathoming the root as a false ``infeasible`` (with
   ``DISCOPT_NARROW_BOX_BRANCH=1`` it did exactly that: ``status="infeasible"``).
"""

import math

import discopt.modeling as dm
import discopt.solver as solver_mod
import numpy as np
import pytest
from discopt._relax.discretization import DiscretizationState
from discopt._relax.milp_relaxation import build_milp_relaxation
from discopt._relax.term_classifier import classify_nonlinear_terms
from discopt._relax.uniform_relax import _pow_curv
from discopt.modeling.core import SolveResult

# --------------------------------------------------------------------------- #
# 2. negative even powers: the relaxation must not cut a feasible point
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize("p", [-2.0, -4.0])
@pytest.mark.parametrize("box", [(-3.0, 3.0), (-1.0, 1.0), (0.0, 3.0), (-2.0, 0.0)])
def test_pow_curv_has_no_verdict_on_a_box_holding_the_pole(p, box):
    assert _pow_curv(p, *box) is None


@pytest.mark.unit
def test_pow_curv_verdicts_that_stand():
    assert _pow_curv(2.0, -3.0, 3.0) == "convex"  # positive even power: all of R
    assert _pow_curv(-2.0, 0.5, 3.0) == "convex"  # pole excluded
    assert _pow_curv(-2.0, -3.0, -0.5) == "convex"
    assert _pow_curv(-1.0, 0.5, 3.0) == "convex"
    assert _pow_curv(-1.0, -3.0, -0.5) == "concave"


@pytest.mark.unit
@pytest.mark.parametrize("p", [-2, -4])
@pytest.mark.parametrize("form", ["pow", "div"], ids=["x**p", "1/x**-p"])
@pytest.mark.parametrize("box", [(-3.0, 3.0), (-1.0, 1.0), (-0.9, 0.7), (-2.0, 1.0)])
def test_relaxation_admits_every_sampled_feasible_point(p, form, box):
    """Feasible-point sampling: ``(x, x**p)`` for sampled ``x != 0`` satisfies every row."""
    cap = 1e4
    m = dm.Model("negpow")
    x = m.continuous("x", lb=box[0], ub=box[1])
    e = x**p if form == "pow" else 1 / x ** (-p)
    m.subject_to(e <= cap)
    m.minimize(e)
    milp, _ = build_milp_relaxation(
        m,
        classify_nonlinear_terms(m),
        DiscretizationState(),
        bound_override=(np.array([box[0]]), np.array([box[1]])),
    )
    n_cols = int(np.size(milp._c))
    assert n_cols >= 2, "expected an aux column for the power atom"
    A = milp._A_ub.toarray() if hasattr(milp._A_ub, "toarray") else np.asarray(milp._A_ub)
    b = np.asarray(milp._b_ub, dtype=float)
    lo_col = np.array([lb for lb, _ in milp._bounds], dtype=float)
    hi_col = np.array([ub for _, ub in milp._bounds], dtype=float)

    xs = np.concatenate([np.linspace(box[0], box[1], 401), [0.5, -0.5, 0.1, -0.1]])
    checked = 0
    for xv in xs:
        if abs(xv) < 1e-12 or xv < box[0] or xv > box[1]:
            continue
        wv = float(xv) ** p
        if wv > cap:
            continue
        # Every aux column after x is this single atom (``w = x**p``) in this model.
        pt = np.full(n_cols, wv)
        pt[0] = xv
        viol = A @ pt - b
        assert np.all(viol <= 1e-7 * (1.0 + np.abs(b))), (
            f"x={xv}: row {int(np.argmax(viol))} violated by {float(viol.max()):.3e}"
        )
        assert np.all(pt >= lo_col - 1e-9) and np.all(pt <= hi_col + 1e-9), f"x={xv}: column box"
        checked += 1
    assert checked > 50, f"only {checked} points were checked"


@pytest.mark.unit
@pytest.mark.parametrize("form", ["pow", "div"], ids=["x**-2", "1/x**2"])
def test_inverse_square_equality_is_solved(form):
    m = dm.Model("t")
    x = m.continuous("x", lb=-3, ub=3)
    m.subject_to((x**-2 if form == "pow" else 1 / x**2) == 4)
    m.maximize(x)
    r = m.solve(time_limit=30)
    assert r.status == "optimal", (r.status, r.error)
    assert float(r.x["x"]) == pytest.approx(0.5, abs=1e-6)
    assert r.bound is not None and r.bound >= 0.5 - 1e-6


@pytest.mark.unit
def test_negative_even_power_optimum_on_the_negative_branch():
    """``x**-4 == 4`` on ``[-0.9, 0.7]``: +0.7071 is outside the box, so -0.7071 wins."""
    m = dm.Model("t")
    x = m.continuous("x", lb=-0.9, ub=0.7)
    m.subject_to(x**-4 == 4)
    m.maximize(x)
    r = m.solve(time_limit=30)
    assert r.status == "optimal", (r.status, r.error)
    assert float(r.x["x"]) == pytest.approx(-1.0 / math.sqrt(2.0), abs=1e-6)


# --------------------------------------------------------------------------- #
# 1. an ``error`` status beside a screened point, and ``error=None``
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_norm_inf_with_a_feasible_point_is_not_reported_as_error():
    m = dm.Model("n")
    v = m.continuous("v", shape=(2,), lb=-2, ub=2)
    m.subject_to(v[0] + v[1] >= 1)
    m.minimize(dm.norm(v, "inf"))
    r = m.solve(time_limit=30)
    assert r.status != "error", f"status=error with x={r.x!r}, error={r.error!r}"
    assert r.status in ("feasible", "optimal")
    vv = np.asarray(r.x["v"], dtype=float)
    assert vv[0] + vv[1] >= 1.0 - 1e-6
    assert r.objective == pytest.approx(float(np.max(np.abs(vv))), abs=1e-9)
    assert r.objective >= 0.5 - 1e-6  # never below the true optimum
    if r.status == "feasible":
        assert r.gap_certified is False


def _infeasible_convex_qcp() -> dm.Model:
    m = dm.Model("inf")
    x = m.continuous("x", lb=-10, ub=10)
    y = m.continuous("y", lb=-10, ub=10)
    m.subject_to(x * x + y * y <= 1.0)
    m.subject_to(x + y >= math.sqrt(2.0) * 1.02)
    m.minimize(x - y)
    return m


@pytest.mark.unit
def test_single_nlp_error_names_the_backend_return_code():
    """The single-NLP route records WHICH failure ``SolveStatus.ERROR`` collapsed."""
    import time

    m = _infeasible_convex_qcp()
    r = solver_mod._solve_continuous(
        m, 20.0, None, time.perf_counter(), "pounce", certify_convex=True
    )
    assert r.status == "error"
    assert r.x is None
    assert r.error is not None and "Infeasible_Problem_Detected" in r.error, r.error


@pytest.mark.unit
def test_error_status_always_carries_a_reason(monkeypatch):
    """Backstop in ``Model.solve``: a route that returns a bare ``error`` gets a reason."""

    def _bare_error(*args, **kwargs):
        return SolveResult(status="error")

    monkeypatch.setattr(solver_mod, "solve_model", _bare_error)
    m = dm.Model("bare")
    x = m.continuous("x", lb=0, ub=1)
    m.minimize(dm.exp(x))
    r = m.solve(time_limit=5)
    assert r.status == "error"
    assert isinstance(r.error, str) and r.error.strip()


@pytest.mark.unit
def test_backstop_keeps_a_route_supplied_reason(monkeypatch):
    def _reasoned_error(*args, **kwargs):
        return SolveResult(status="error", error="specific cause")

    monkeypatch.setattr(solver_mod, "solve_model", _reasoned_error)
    m = dm.Model("reasoned")
    x = m.continuous("x", lb=0, ub=1)
    m.minimize(dm.exp(x))
    assert m.solve(time_limit=5).error == "specific cause"
