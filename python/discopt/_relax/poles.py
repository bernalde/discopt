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

Only *detection* lives here. It changes no bound, no box and no status. The one
consumer that acts on it, ``DISCOPT_POLE_BRANCHING`` (:func:`objective_pole_loci`),
uses it only to choose WHERE the B&B tree splits an unbounded node.
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
    Variable,
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


@dataclass(frozen=True)
class PoleLocus:
    """A hyperplane ``x[col] == value`` on which an objective denominator vanishes."""

    col: int
    value: float


def _column_offsets(model: Model) -> dict[int, int]:
    offsets: dict[int, int] = {}
    off = 0
    for v in model._variables:
        offsets[id(v)] = off
        off += int(v.size)
    return offsets


def _flat_column(expr: Expression, offsets: dict[int, int]) -> Optional[int]:
    """Flat column of a scalar variable reference, or ``None``."""
    if isinstance(expr, Variable):
        return offsets.get(id(expr)) if int(expr.size) == 1 else None
    if isinstance(expr, IndexExpression) and isinstance(expr.base, Variable):
        base = expr.base
        idx = expr.index
        idx_t = idx if isinstance(idx, tuple) else (idx,)
        if not all(isinstance(i, (int, np.integer)) for i in idx_t):
            return None
        shape = tuple(base.shape)
        if len(idx_t) != len(shape):
            return None
        try:
            flat = int(np.ravel_multi_index(tuple(int(i) for i in idx_t), shape))
        except ValueError:
            return None
        start = offsets.get(id(base))
        return None if start is None else start + flat
    return None


# An affine form in at most one scalar column: ``(col, a, b)`` means
# ``a * x[col] + b``; ``col is None`` means the constant ``b``.
_Affine = tuple[Optional[int], float, float]


def _affine1(expr: Expression, offsets: dict[int, int]) -> Optional[_Affine]:
    if isinstance(expr, Constant):
        v = np.asarray(expr.value, dtype=np.float64)
        return (None, 0.0, float(v)) if v.size == 1 else None
    col = _flat_column(expr, offsets)
    if col is not None:
        return (col, 1.0, 0.0)
    if isinstance(expr, UnaryOp) and expr.op == "neg":
        f = _affine1(expr.operand, offsets)
        return None if f is None else (f[0], -f[1], -f[2])
    if not isinstance(expr, BinaryOp) or expr.op not in ("+", "-", "*", "/"):
        return None
    left = _affine1(expr.left, offsets)
    right = _affine1(expr.right, offsets)
    if left is None or right is None:
        return None
    (lc, la, lb), (rc, ra, rb) = left, right
    if expr.op in ("+", "-"):
        s = 1.0 if expr.op == "+" else -1.0
        if lc is not None and rc is not None and lc != rc:
            return None
        return (lc if lc is not None else rc, la + s * ra, lb + s * rb)
    if expr.op == "*":
        if lc is None:
            return (rc, lb * ra, lb * rb)
        if rc is None:
            return (lc, rb * la, rb * lb)
        return None
    # "/": only by a nonzero constant
    if rc is None and rb != 0.0:
        return (lc, la / rb, lb / rb)
    return None


def _zeros(den: Expression, offsets: dict[int, int]) -> list[PoleLocus]:
    """Hyperplanes on which ``den`` vanishes, where they can be located exactly."""
    aff = _affine1(den, offsets)
    if aff is not None:
        col, a, b = aff
        if col is None or a == 0.0:
            return []
        root = -b / a
        return [PoleLocus(col, root)] if np.isfinite(root) else []
    if isinstance(den, BinaryOp) and den.op == "**" and isinstance(den.right, Constant):
        p = np.asarray(den.right.value)
        if p.ndim == 0 and float(p) > 0.0:
            return _zeros(den.left, offsets)
    if isinstance(den, BinaryOp) and den.op == "*":
        return _zeros(den.left, offsets) + _zeros(den.right, offsets)
    if isinstance(den, UnaryOp) and den.op in ("neg", "abs"):
        return _zeros(den.operand, offsets)
    return []


def objective_pole_loci(model: Model) -> list[PoleLocus]:
    """Where each objective denominator vanishes, as flat ``(column, value)`` pairs.

    Used by ``DISCOPT_POLE_BRANCHING`` (#1493) to put a spatial split exactly on
    a pole, so each child box sees a one-signed denominator. A denominator that
    is not an affine function of one scalar column (or a positive power /
    product of such) contributes nothing -- the tree then bisects. Only WHERE
    to branch comes from here; no bound, box or status reads it.
    """
    obj = getattr(model, "_objective", None)
    if obj is None or getattr(obj, "expression", None) is None:
        return []
    offsets = _column_offsets(model)
    out: list[PoleLocus] = []
    for _term, den in _walk(obj.expression):
        for locus in _zeros(den, offsets):
            if locus not in out:
                out.append(locus)
    return out


def pole_branch_point(loci: list[PoleLocus], lb: np.ndarray, ub: np.ndarray) -> Optional[PoleLocus]:
    """Where to split an unbounded node box ``[lb, ub]``, or ``None``.

    The first locus strictly inside the box: a split there leaves the pole on
    each child's boundary. Failing that, a locus on the box's boundary names the
    column the objective diverges along, so bisect THAT column (the midpoint of
    its range) rather than the longest edge: the child away from the pole can
    then be bounded, and the one touching it shrinks toward it geometrically
    until the tree's depth cap ends the chain. (The tree treats a hinted column
    it can no longer split as exhausted and stops: splitting any other column
    cannot remove a pole that an affine-in-one-column denominator puts there.)
    """
    for p in loci:
        if p.col < len(lb) and lb[p.col] < p.value < ub[p.col]:
            return p
    for p in loci:
        if p.col < len(lb) and lb[p.col] <= p.value <= ub[p.col]:
            lo, hi = float(lb[p.col]), float(ub[p.col])
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                return PoleLocus(p.col, 0.5 * (lo + hi))
            return p
    return None


def describe_poles(poles: list[ObjectivePole]) -> str:
    """One human-readable clause per pole, for the no-bound diagnostic."""
    return "; ".join(
        f"`{p.term}` divides by `{p.denominator}`, whose range [{p.lo:.6g}, {p.hi:.6g}] contains 0"
        for p in poles
    )
