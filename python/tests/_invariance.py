"""Metamorphic invariance helpers (#1537 workstream B).

A certified answer is a statement about the model, not about its coordinates. Two
changes of variables leave every model mathematically identical, and a global
solver must certify the same optimum before and after them:

* **translation** ``x = y - c``: each variable is replaced by a fresh one whose box
  is shifted by ``c`` (an integer shift for integer variables, so integrality is
  preserved), and every occurrence of ``x`` in the objective and the rows becomes
  ``y - c``;
* **row rescaling**: every constraint body is multiplied by a positive ``s``.

These are instruments, so they refuse rather than guess (CLAUDE.md §6/§7): an
expression node the substitution does not know raises ``TypeError`` instead of
being passed through untranslated -- a silently untranslated sub-expression would
make the "shifted" model a different model and the comparison meaningless.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Constraint,
    Expression,
    FunctionCall,
    IndexExpression,
    MatMulExpression,
    Model,
    Objective,
    Parameter,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
    VarType,
)


def substitute(expr: Expression, var_map: dict[int, Expression]) -> Expression:
    """``expr`` with every Variable ``v`` (keyed by ``id(v)``) replaced by ``var_map``.

    Raises ``TypeError`` on any node type it does not rebuild explicitly.
    """
    if isinstance(expr, Variable):
        return var_map[id(expr)]
    if isinstance(expr, (Constant, Parameter)):
        return expr
    if isinstance(expr, IndexExpression):
        return IndexExpression(substitute(expr.base, var_map), expr.index)
    if isinstance(expr, BinaryOp):
        return BinaryOp(expr.op, substitute(expr.left, var_map), substitute(expr.right, var_map))
    if isinstance(expr, UnaryOp):
        return UnaryOp(expr.op, substitute(expr.operand, var_map))
    if type(expr) is FunctionCall:
        return FunctionCall(expr.func_name, *(substitute(a, var_map) for a in expr.args))
    if isinstance(expr, MatMulExpression):
        return MatMulExpression(substitute(expr.left, var_map), substitute(expr.right, var_map))
    if isinstance(expr, SumExpression):
        return SumExpression(substitute(expr.operand, var_map), axis=expr.axis)
    if isinstance(expr, SumOverExpression):
        return SumOverExpression([substitute(t, var_map) for t in expr.terms])
    raise TypeError(f"invariance harness: no substitution rule for {type(expr).__name__}")


def _rebuild(
    model: Model,
    shift_of: Callable[[Variable], np.ndarray],
    row_scale: float,
    name: str,
) -> Model:
    """A fresh model whose variable ``v`` is ``y - shift_of(v)`` and whose rows are
    each multiplied by ``row_scale``. Refuses models it cannot rebuild faithfully."""
    if model._objective is None:
        raise ValueError("invariance harness: model has no objective")
    # Indicator / disjunctive / SOS / logical rows live in ``_constraints`` as their
    # own classes; only plain algebraic rows are rebuilt, everything else refuses.
    for con in model._constraints:
        if type(con) is not Constraint:
            raise ValueError(f"invariance harness: cannot rebuild a {type(con).__name__} row")
    # ``_builder_linear_blocks`` holds the fast-path rows (``constraint(fast=True)``
    # / ``add_linear_constraints``), which never enter ``_constraints``: rebuilding
    # without them would silently drop rows (#1546 review).
    for attr in (
        "_piecewise_domains",
        "_complementarities",
        "_lowered_complementarities",
        "_builder_linear_blocks",
    ):
        if getattr(model, attr, None):
            raise ValueError(f"invariance harness: cannot rebuild a model with {attr}")
    if not row_scale > 0.0:
        raise ValueError("row_scale must be positive (a negative one flips every sense)")
    new = Model(name)
    var_map: dict[int, Expression] = {}
    shifts: list[np.ndarray] = []
    for v in model._variables:
        c = np.broadcast_to(np.asarray(shift_of(v), dtype=np.float64), v.lb.shape)
        shifts.append(np.ravel(c))
        if v.var_type is not VarType.CONTINUOUS and np.any(c != np.round(c)):
            raise ValueError(f"integer variable {v.name} needs an integer shift")
        lb = np.where(np.isfinite(v.lb), v.lb + c, v.lb)
        ub = np.where(np.isfinite(v.ub), v.ub + c, v.ub)
        shape = v.shape if v.shape else ()
        if v.var_type is VarType.CONTINUOUS:
            y = new.continuous(v.name, shape=shape, lb=lb, ub=ub)
        elif v.var_type is VarType.BINARY and not np.any(c):
            y = new.binary(v.name, shape=shape)
        else:  # a shifted binary is an integer on {c, c+1}
            y = new.integer(v.name, shape=shape, lb=lb, ub=ub)
        var_map[id(v)] = y - Constant(c if shape else float(c)) if np.any(c) else y
    for con in model._constraints:
        body = substitute(con.body, var_map)
        if row_scale != 1.0:
            body = row_scale * body
        new._constraints.append(Constraint(body=body, sense=con.sense, rhs=0.0, name=con.name))
    new._objective = Objective(
        expression=substitute(model._objective.expression, var_map),
        sense=model._objective.sense,
    )
    # Flat shift vector in ``_variables`` order: a point ``x`` of ``model`` is the
    # point ``x + shift`` of ``new`` (used by the harness's own self-test).
    new._invariance_shift = np.concatenate(shifts) if shifts else np.zeros(0)
    return new


def translate(model: Model, offset: float, *, seed: int = 0) -> Model:
    """``model`` under ``x = y - c``, with each variable's ``c`` drawn around ``offset``
    (rounded to an integer, sign random) so different columns sit at different places."""
    rng = np.random.default_rng(seed)

    def shift_of(v: Variable) -> np.ndarray:
        size = v.lb.shape
        sign = rng.choice([-1.0, 1.0], size=size)
        return np.round(sign * offset * rng.uniform(0.5, 1.5, size=size))

    return _rebuild(model, shift_of, 1.0, f"{model.name}_shift{offset:g}")


def rescale_rows(model: Model, scale: float) -> Model:
    """``model`` with every constraint body multiplied by ``scale`` (> 0)."""
    return _rebuild(model, lambda v: np.zeros(v.lb.shape), scale, f"{model.name}_rows{scale:g}")


def certified_answer_changed(base, other, *, rel: float = 1e-4, abs_tol: float = 1e-6) -> str:
    """'' when ``other`` is consistent with the certified ``base``; else a reason.

    Only a certificate is compared: an uncertified ``other`` is a *lost*
    certificate (reported separately by the caller), never a false one.
    """
    if not other.gap_certified:
        return ""
    if base.status == "infeasible" and base.gap_certified:
        if other.status == "infeasible":
            return ""
        return f"certified {other.status} on an infeasible model"
    if not (base.status == "optimal" and base.gap_certified):
        return ""
    if other.status == "infeasible":
        return "certified infeasible on a feasible model"
    tol = max(abs_tol, rel * max(1.0, abs(base.objective)))
    if abs(other.objective - base.objective) > tol:
        return f"certified {other.objective!r}, base certified {base.objective!r}"
    return ""


def invariance_violation(
    base, other, original: Model, transformed: Model, *, rel: float = 1e-4, abs_tol: float = 1e-6
) -> str:
    """'' when the transformed solve's result is consistent with the certified
    ``base`` of the original model; else the reason. Checks every result, not
    only certified ones (#1546 review):

    * a certificate must agree with the base (:func:`certified_answer_changed`);
    * a published dual bound must not cross the base optimum, certified or not
      (an uncertified result can still publish a false bound);
    * a published incumbent, mapped back to the original coordinates, must be
      feasible for the ORIGINAL model and must not beat the base optimum.
    """
    from discopt.modeling.core import ObjectiveSense
    from discopt.validation.feasibility import verify_point

    why = certified_answer_changed(base, other, rel=rel, abs_tol=abs_tol)
    if why or not (base.status == "optimal" and base.gap_certified):
        return why
    tol = max(abs_tol, rel * max(1.0, abs(base.objective)))
    maximize = original._objective.sense is ObjectiveSense.MAXIMIZE
    if other.bound is not None and np.isfinite(other.bound):
        crosses = (
            other.bound < base.objective - tol if maximize else other.bound > base.objective + tol
        )
        if crosses:
            return f"bound {other.bound!r} crosses the base optimum {base.objective!r}"
    if other.x is not None and other.status in ("optimal", "feasible"):
        flat = np.concatenate(
            [
                np.ravel(np.asarray(other.x[v.name], dtype=np.float64))
                for v in transformed._variables
            ]
        )
        x = flat - transformed._invariance_shift
        check = verify_point(original, x, with_objective=True)
        if not check.ok:
            return f"published incumbent is infeasible in the original model ({check.reason})"
        better = (
            check.objective > base.objective + tol
            if maximize
            else check.objective < base.objective - tol
        )
        if better:
            return (
                f"published incumbent {check.objective!r} beats the certified base "
                f"{base.objective!r}"
            )
    return ""
