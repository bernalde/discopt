"""Poles of the objective inside the declared box (#1493).

``min 1/x`` over ``x in [-5, 5]`` has no dual bound on any box that contains
``x = 0``: the objective is unbounded below as ``x -> 0-``. The relaxation layer
correctly refuses to bound it -- the ratio envelope keeps an infinite interval
floor -- so the spatial B&B stops with a feasible point and no bound. That exit
is honest under :mod:`discopt.status` (state 3, "feasible, no bound"; ``unbounded``
means *certified* unboundedness, which a relaxation cannot prove at a pole), but
the diagnostic it produced was not: it blamed "a nonlinear term with no envelope"
and advised an epigraph reformulation, which cannot help at a pole. This module
finds the actual cause so the solve can say it.

Only *detection* lives here. It changes no bound, no box and no status.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Expression,
    FunctionCall,
    IndexExpression,
    MatMulExpression,
    Model,
    SumExpression,
    SumOverExpression,
    UnaryOp,
)


@dataclass(frozen=True)
class ObjectivePole:
    """One singular objective subterm: ``term`` divides by ``denominator``."""

    term: str
    denominator: str
    lo: float
    hi: float


def _children(expr: Expression) -> list[Expression]:
    if isinstance(expr, BinaryOp):
        return [expr.left, expr.right]
    if isinstance(expr, UnaryOp):
        return [expr.operand]
    if isinstance(expr, FunctionCall):
        return list(expr.args)
    if isinstance(expr, IndexExpression):
        return [expr.base]
    if isinstance(expr, SumExpression):
        return [expr.operand]
    if isinstance(expr, SumOverExpression):
        return list(expr.terms)
    if isinstance(expr, MatMulExpression):
        return [expr.left, expr.right]
    return []


def _denominator(expr: Expression) -> Optional[Expression]:
    """The expression whose zero is a pole of ``expr``, or ``None``."""
    if not isinstance(expr, BinaryOp):
        return None
    if expr.op == "/" and not isinstance(expr.right, Constant):
        return expr.right
    if expr.op == "**" and isinstance(expr.right, Constant):
        p = np.asarray(expr.right.value)
        if p.ndim == 0 and float(p) < 0.0 and not isinstance(expr.left, Constant):
            return expr.left
    return None


def _walk(expr: Expression) -> Iterator[tuple[Expression, Expression]]:
    seen: set[int] = set()
    stack = [expr]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        den = _denominator(node)
        if den is not None:
            yield node, den
        stack.extend(_children(node))


def objective_poles(model: Model, limit: int = 3) -> list[ObjectivePole]:
    """Objective subterms whose denominator's range over the DECLARED box holds 0.

    Uses the sound interval enclosure of each denominator; a denominator whose
    enclosure cannot be computed as a scalar is skipped (never guessed).
    """
    from discopt._relax.convexity.interval_eval import evaluate_interval

    obj = getattr(model, "_objective", None)
    if obj is None or getattr(obj, "expression", None) is None:
        return []
    out: list[ObjectivePole] = []
    for term, den in _walk(obj.expression):
        enc = evaluate_interval(den, model)
        lo_arr = np.asarray(enc.lo, dtype=np.float64)
        hi_arr = np.asarray(enc.hi, dtype=np.float64)
        if lo_arr.size != 1 or hi_arr.size != 1:
            continue
        lo = float(lo_arr.reshape(()))
        hi = float(hi_arr.reshape(()))
        if lo <= 0.0 <= hi:
            out.append(ObjectivePole(repr(term), repr(den), lo, hi))
            if len(out) >= limit:
                break
    return out


def describe_poles(poles: list[ObjectivePole]) -> str:
    """One human-readable clause per pole, for the no-bound diagnostic."""
    return "; ".join(
        f"`{p.term}` divides by `{p.denominator}`, whose range [{p.lo:.6g}, {p.hi:.6g}] contains 0"
        for p in poles
    )
