"""#1561 -- a certified incumbent must pass ``verify_point`` on the solved model.

The defect
----------

``tls2`` with every row multiplied by 1e6 was returned ``optimal`` with
``gap_certified=True`` on an incumbent that ``verify_point`` rejects: row 18
(``x2 - 3 x31 - 8 x32 - 15 x33 - 1 == 0``, scaled) violated by 6.669 where 1.0 is
allowed.

Root cause: the NLP-BB route's C-3 integer snap is refused when the snapped point
leaves the rows, and the route then reported the UNROUNDED point. Its exit gate
judged that point as computed, integer columns up to 1e-5 off an integer, and the
rows it satisfies are satisfied with slack bought by that fractionality.
``verify_point`` judges the integral realisation (#1380), so the two arbiters
disagreed: x31 off by ~2.2e-6 times the row's coefficient 3 (times 1e6) is the
6.669.

The two layers, and why both
----------------------------

1. **The route.** NLP-BB now judges the incumbent's integral realisation at the
   refine-adoption step (so a refine with integers pinned exactly beats a
   fractional incumbent whose snap fails) and at the exit gate; a point whose
   integral realisation fails is reported uncertified.
2. **The class.** ``Model.solve`` runs ``verify_point`` -- on the pre-solve
   snapshot of the declared rows -- on every certified result before handing it
   back, on the default path and the convex-kernel early return. A route whose
   gate drifts from ``verify_point`` again cannot certify what it rejects.

The fast tests below pin layer 2 deterministically by publishing a certified
point from a stubbed route. The ``slow`` test runs the real tls2 x1e6 solve; that
one is timing-dependent (the bad incumbent appeared on a loaded runner, see the
issue), so it asserts the invariant rather than a particular path.
"""

from __future__ import annotations

import os
from pathlib import Path

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling.core import SolveResult
from discopt.validation.feasibility import verify_point

DATA = Path(__file__).parent / "data" / "minlplib_nl"

#: ``y`` is integer; ``y = 1 + 3e-6`` is inside the 1e-5 integrality tolerance and
#: satisfies ``x - 3 y == 1`` exactly with ``x = 4.000009``. Its integral
#: realisation ``y = 1`` leaves the row violated by 9e-6, where ``verify_point``
#: allows ``1e-6 * max(1, |J x|) = 4e-6``.
FRAC_Y = 1.0 + 3e-6
FRAC_X = 1.0 + 3.0 * FRAC_Y


def _model(nonlinear: bool):
    m = dm.Model("frac_certificate")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.integer("y", lb=0, ub=3)
    m.subject_to(x - 3 * y == 1, name="link")
    if nonlinear:
        # A nonlinear row puts the model outside the LP/MILP fast family, so
        # ``Model.solve`` takes its pre-solve snapshot evaluator (the arm under
        # test); the linear control below exercises the no-snapshot arm.
        m.subject_to(dm.exp(x) <= 1e3, name="nl")
    m.minimize(x + y)
    return m


def _certified(x, y, *, bound=4.0):
    r = SolveResult(
        status="optimal",
        objective=float(x + y),
        bound=bound,
        gap=0.0,
        x={"x": np.array(x), "y": np.array(y)},
        gap_certified=True,
    )
    r._set_bound(bound, valid=True, source="bnb_tree")
    return r


@pytest.fixture
def no_kernel(monkeypatch):
    """Keep the convex kernel out of the way, so ``solve_model`` is what returns."""
    import discopt.solvers._convex_kernel as ck

    monkeypatch.setattr(ck, "try_convex_solve", lambda *a, **k: None)


def _publish(monkeypatch, result):
    import discopt.solver as solver

    monkeypatch.setattr(solver, "solve_model", lambda model, **kw: result)


def test_the_fixture_point_fails_verify_point():
    """The probe fired: the published point is one ``verify_point`` rejects, its
    integral realisation is what fails, and the integral control passes."""
    m = _model(nonlinear=True)
    bad = verify_point(m, np.array([FRAC_X, FRAC_Y]))
    assert not bad.ok, bad
    assert "row 0" in str(bad.reason), bad
    assert verify_point(m, np.array([4.0, 1.0])).ok


@pytest.mark.parametrize("nonlinear", [True, False], ids=["snapshot", "no-snapshot"])
def test_a_certified_point_that_fails_verify_point_is_decertified(
    monkeypatch, no_kernel, nonlinear
):
    m = _model(nonlinear)
    _publish(monkeypatch, _certified(FRAC_X, FRAC_Y))
    res = m.solve(time_limit=5.0)

    assert res.status == "feasible", res.status
    assert res.gap_certified is False
    assert (res.solver_stats or {}).get("certificate/incumbent_unverified") == 1.0
    # Not a withhold: the loose #772 screen passes this point; only the
    # certificate is withdrawn. The dual bound stands.
    assert res.x is not None and res.objective is not None
    assert res.bound == pytest.approx(4.0)


@pytest.mark.parametrize("nonlinear", [True, False], ids=["snapshot", "no-snapshot"])
def test_an_integral_certified_point_keeps_its_certificate(monkeypatch, no_kernel, nonlinear):
    m = _model(nonlinear)
    _publish(monkeypatch, _certified(4.0, 1.0))
    res = m.solve(time_limit=5.0)

    assert res.status == "optimal" and res.gap_certified is True
    assert "certificate/incumbent_unverified" not in (res.solver_stats or {})


def test_the_convex_kernel_return_is_checked_too(monkeypatch):
    """The kernel returns before ``solve_model`` and the #772 screen; the
    backstop must run on that return as well."""
    import discopt.solvers._convex_kernel as ck

    m = _model(nonlinear=True)
    bad = _certified(FRAC_X, FRAC_Y)
    monkeypatch.setattr(ck, "try_convex_solve", lambda *a, **k: bad)
    res = m.solve(time_limit=5.0)

    assert res is bad, "the stub did not take the kernel's early return"
    assert res.status == "feasible" and res.gap_certified is False
    assert (res.solver_stats or {}).get("certificate/incumbent_unverified") == 1.0


def test_verify_point_judges_the_rows_of_the_evaluator_it_is_given():
    """``verify_point(..., evaluator=)`` is what lets ``Model.solve`` judge the
    pre-solve rows: the rows come from that evaluator, not from ``model``."""
    from discopt._tape_nlp_evaluator import make_evaluator

    looser = dm.Model("looser")
    lx = looser.continuous("x", lb=0.0, ub=10.0)
    ly = looser.integer("y", lb=0, ub=3)
    looser.subject_to(lx - 3 * ly <= 10, name="link")
    looser.subject_to(dm.exp(lx) <= 1e3, name="nl")
    looser.minimize(lx + ly)

    m = _model(nonlinear=True)
    x = np.array([FRAC_X, FRAC_Y])
    assert not verify_point(m, x).ok
    assert verify_point(m, x, evaluator=make_evaluator(looser)).ok


# ── the reported instance ───────────────────────────────────────────────────


def _substitute(expr, var_map):
    from discopt.modeling.core import (
        BinaryOp,
        Constant,
        FunctionCall,
        IndexExpression,
        MatMulExpression,
        Parameter,
        SumExpression,
        SumOverExpression,
        UnaryOp,
        Variable,
    )

    if isinstance(expr, Variable):
        return var_map[id(expr)]
    if isinstance(expr, (Constant, Parameter)):
        return expr
    if isinstance(expr, IndexExpression):
        return IndexExpression(_substitute(expr.base, var_map), expr.index)
    if isinstance(expr, BinaryOp):
        return BinaryOp(expr.op, _substitute(expr.left, var_map), _substitute(expr.right, var_map))
    if isinstance(expr, UnaryOp):
        return UnaryOp(expr.op, _substitute(expr.operand, var_map))
    if type(expr) is FunctionCall:
        return FunctionCall(expr.func_name, *(_substitute(a, var_map) for a in expr.args))
    if isinstance(expr, MatMulExpression):
        return MatMulExpression(_substitute(expr.left, var_map), _substitute(expr.right, var_map))
    if isinstance(expr, SumExpression):
        return SumExpression(_substitute(expr.operand, var_map), axis=expr.axis)
    if isinstance(expr, SumOverExpression):
        return SumOverExpression([_substitute(t, var_map) for t in expr.terms])
    raise TypeError(f"no substitution rule for {type(expr).__name__}")


def _rescale_rows(model, scale):
    """A fresh copy of ``model`` with every row body multiplied by ``scale``
    (the #1537 invariance harness's row transform). Refuses rather than guesses."""
    from discopt.modeling.core import Constraint, Model, Objective, VarType

    assert all(type(c) is Constraint for c in model._constraints)
    new = Model(f"{model.name}_rows{scale:g}")
    var_map = {}
    for v in model._variables:
        shape = v.shape if v.shape else ()
        if v.var_type is VarType.CONTINUOUS:
            y = new.continuous(v.name, shape=shape, lb=v.lb, ub=v.ub)
        elif v.var_type is VarType.BINARY:
            y = new.binary(v.name, shape=shape)
        else:
            y = new.integer(v.name, shape=shape, lb=v.lb, ub=v.ub)
        var_map[id(v)] = y
    for con in model._constraints:
        body = scale * (_substitute(con.body, var_map) - con.rhs)
        new._constraints.append(Constraint(body=body, sense=con.sense, rhs=0.0, name=con.name))
    new._objective = Objective(
        expression=_substitute(model._objective.expression, var_map),
        sense=model._objective.sense,
    )
    return new


@pytest.mark.slow
@pytest.mark.parametrize("scale", [1.0, 1e6])
def test_tls2_certified_incumbent_passes_verify_point(scale):
    m = dm.from_nl(str(DATA / "tls2.nl"))
    t = _rescale_rows(m, scale) if scale != 1.0 else m
    res = t.solve(time_limit=float(os.environ.get("DISCOPT_1561_TL", "20")))
    assert res.x, f"no incumbent ({res.status})"
    x = np.concatenate(
        [np.atleast_1d(np.asarray(res.x[v.name], float)).ravel() for v in t._variables]
    )
    verdict = verify_point(t, x)
    if res.gap_certified or res.status == "optimal":
        assert verdict.ok, (
            f"certified {res.status} (objective {res.objective!r}) on a point "
            f"verify_point rejects: {verdict.reason}"
        )
        # tls2's optimum is 5.3 (minlplib.solu); rows x1e6 do not move it.
        assert res.objective == pytest.approx(5.3, rel=1e-4)
