"""#1543 on real MINLPLib instances that actually exercise the origin shift.

The synthetic tests in ``test_1543_miqp_origin_shift.py`` pin the mechanism; this
file pins it on the real corpus. The instances were chosen by instrumentation, not
by name. Every in-repo MINLPLib ``.nl`` and QPLIB instance (131 distinct) was
solved with spies on ``_solve_miqp_bb`` and ``_miqp_origin_shift``:

* unshifted, only ``alan`` and ``QPLIB_3871`` reach ``_solve_miqp_bb``, and
  both with a zero shift, so the fix is a no-op on the corpus as shipped;
* under the #1537 translation ``x = y - c`` (offsets 1e3/1e5/1e6, 393 runs,
  none refused by the transform), these reach it with a nonzero shift:
  ``alan, meanvarx, st_miqp1..5, st_test1, st_testgr3`` (and ``QPLIB_3871``,
  which no route certifies at 60 s, shifted or not, so it has no certified
  reference to test against).

Shift ON vs OFF (OFF = ``_miqp_origin_shift`` patched to ``None``, which is the
``main`` path; 25/30 rows byte-identical to a ``main`` worktree, and the other 5
are 60 s time-limit exits whose node counts differ by wall clock): ON certified
27/30 with zero false certificates. OFF certified 14/30, of which 5 were false
(``st_miqp3`` 12 and 0 vs -6, ``st_testgr3`` -20.17 and -20.00 vs -20.59,
``meanvarx`` 14.549 vs 14.369), and it raised the #952 refusal 5 times.

``_translate`` is #1546's ``python/tests/_invariance.translate`` (same seeding,
so the cases are the measured ones), reduced to what these models need; switch
to the shared helper once #1546 lands.
"""

from __future__ import annotations

from pathlib import Path

import discopt.solver as solver_mod
import numpy as np
import pytest
from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Constraint,
    Expression,
    FunctionCall,
    IndexExpression,
    Model,
    Objective,
    Parameter,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
    VarType,
    from_nl,
)

_DATA = Path(__file__).parent / "data"


def _substitute(expr: Expression, var_map: dict[int, Expression]) -> Expression:
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
    if isinstance(expr, SumExpression):
        return SumExpression(_substitute(expr.operand, var_map), axis=expr.axis)
    if isinstance(expr, SumOverExpression):
        return SumOverExpression([_substitute(t, var_map) for t in expr.terms])
    # An untranslated sub-expression would make the "shifted" model a different
    # model; refuse rather than pass it through (CLAUDE.md §6/§7).
    raise TypeError(f"no substitution rule for {type(expr).__name__}")


def _translate(model: Model, offset: float, seed: int = 0) -> Model:
    """``model`` under ``x = y - c`` with integer ``c`` drawn around ``offset``."""
    rng = np.random.default_rng(seed)
    assert all(type(con) is Constraint for con in model._constraints)
    new = Model(f"{model.name}_shift{offset:g}")
    var_map: dict[int, Expression] = {}
    for v in model._variables:
        size = v.lb.shape
        sign = rng.choice([-1.0, 1.0], size=size)
        c = np.round(sign * offset * rng.uniform(0.5, 1.5, size=size))
        lb = np.where(np.isfinite(v.lb), v.lb + c, v.lb)
        ub = np.where(np.isfinite(v.ub), v.ub + c, v.ub)
        shape = v.shape if v.shape else ()
        if v.var_type is VarType.CONTINUOUS:
            y = new.continuous(v.name, shape=shape, lb=lb, ub=ub)
        else:  # a shifted binary is an integer on {c, c + 1}
            y = new.integer(v.name, shape=shape, lb=lb, ub=ub)
        var_map[id(v)] = y - Constant(c if shape else float(c))
    for con in model._constraints:
        new._constraints.append(
            Constraint(body=_substitute(con.body, var_map), sense=con.sense, rhs=0.0, name=con.name)
        )
    assert model._objective is not None
    new._objective = Objective(
        expression=_substitute(model._objective.expression, var_map),
        sense=model._objective.sense,
    )
    return new


def _path(name: str) -> Path:
    for sub in ("minlplib_nl", "minlplib"):
        p = _DATA / sub / name
        if p.exists():
            return p
    raise FileNotFoundError(name)


@pytest.fixture
def route_spy(monkeypatch):
    """Prove the run exercised the change: the route ran AND the shift was nonzero.

    Pins ``DISCOPT_RECENTRE=0``: this file measures ``_miqp_origin_shift`` on the
    model AS TRANSLATED. Recentring (default ON since #1537) moves these boxes back
    to the origin before the MIQP route sees them, so the origin shift has nothing
    left to do and the spy's premise fails. The recentred arm is pinned separately
    by ``test_translated_real_instance_certifies_under_recentring`` below."""
    monkeypatch.setenv("DISCOPT_RECENTRE", "0")
    seen = {"bb": 0, "shift_cols": 0}
    real_bb, real_sh = solver_mod._solve_miqp_bb, solver_mod._miqp_origin_shift

    def bb(*a, **k):
        seen["bb"] += 1
        return real_bb(*a, **k)

    def sh(*a, **k):
        out = real_sh(*a, **k)
        if out is not None:
            seen["shift_cols"] = max(seen["shift_cols"], int(np.count_nonzero(out[0])))
        return out

    monkeypatch.setattr(solver_mod, "_solve_miqp_bb", bb)
    monkeypatch.setattr(solver_mod, "_miqp_origin_shift", sh)
    return seen


# Selected by the scan described in the module docstring. All 27 cases together
# take ~10 s (slowest: st_testgr3 at 1.4 s), so none is marked ``slow``.
_CASES = [
    (name, off)
    for name in (
        "alan.nl",
        "meanvarx.nl",
        "st_miqp1.nl",
        "st_miqp2.nl",
        "st_miqp3.nl",
        "st_miqp4.nl",
        "st_miqp5.nl",
        "st_test1.nl",
        "st_testgr3.nl",
    )
    for off in (1e3, 1e5, 1e6)
]


@pytest.mark.parametrize("name, offset", _CASES)
def test_translated_real_instance_keeps_its_certificate(name, offset, route_spy):
    base = from_nl(str(_path(name)))
    ref = base.solve(time_limit=60)
    assert ref.status == "optimal" and ref.gap_certified, (name, ref.status)

    route_spy.update(bb=0, shift_cols=0)
    r = _translate(base, offset).solve(time_limit=60)
    assert route_spy["bb"] >= 1, "did not reach _solve_miqp_bb: the change was not exercised"
    assert route_spy["shift_cols"] > 0, "zero shift: the change was not exercised"

    tol = max(1e-6, 1e-4 * max(1.0, abs(ref.objective)))
    assert r.status == "optimal" and r.gap_certified, (r.status, r.objective, r.bound)
    assert abs(r.objective - ref.objective) <= tol, (r.objective, ref.objective)
    assert r.bound <= ref.objective + tol, (r.bound, ref.objective)


# ``st_testgr3`` at 1e3 is excluded because recentring moves none of its columns:
# its boxes are wider than 1e3 / ratio, so the solve equals the OFF arm above and
# would test nothing new (measured: ``recentre/variables_moved`` absent).
_ON_CASES = [case for case in _CASES if case != ("st_testgr3.nl", 1e3)]


@pytest.mark.parametrize("name, offset", _ON_CASES)
def test_translated_real_instance_certifies_under_recentring(name, offset, monkeypatch):
    """The same 27 translated instances with recentring ON (the default): the
    pass moves the offset columns and the solve still certifies the reference
    optimum. Measured 26/26 certified, 0 false (PR #1594, round 2)."""
    monkeypatch.setenv("DISCOPT_RECENTRE", "1")
    base = from_nl(str(_path(name)))
    ref = base.solve(time_limit=60)
    assert ref.status == "optimal" and ref.gap_certified, (name, ref.status)

    r = _translate(base, offset).solve(time_limit=60)
    moved = (r.solver_stats or {}).get("recentre/variables_moved")
    assert moved is not None and moved > 0, "recentring did not move anything: not exercised"

    tol = max(1e-6, 1e-4 * max(1.0, abs(ref.objective)))
    assert r.status == "optimal" and r.gap_certified, (r.status, r.objective, r.bound)
    assert abs(r.objective - ref.objective) <= tol, (r.objective, ref.objective)
    assert r.bound <= ref.objective + tol, (r.bound, ref.objective)
