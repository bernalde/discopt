"""Exact recentring of large-offset variables before a solve (#1537 workstream C).

A variable whose box sits far from the origin -- ``x in [1e6, 1e6 + 3]`` -- is the
same model as ``x = z + 1e6, z in [0, 3]``, but not the same *numerics*: rows with
activities of ~1e6, objectives whose constant-free part is ~1e6 at every feasible
point, and polynomial expansions with ~c^k terms. The invariance harness
(``python/tests/test_1537_invariance.py``) measured the cost: under ``x = y - 1e6``
the corpus lost 23 of 69 certificates and four models came back with FALSE ones
(#1542, #1543), plus two crashes (#1544).

This module rebuilds such a model in recentred coordinates and maps the result
back. It is an exact change of variables -- no bound, row or objective is
approximated -- so the certified answer of the recentred model is the certified
answer of the original.

Two requirements came out of the entry experiment, both binding:

* **Constant folding.** Substituting ``x = z + c`` alone leaves ``(z + c) - c``
  wherever the user wrote ``x - c``. On ``st_e36`` that unfolded form certified
  *infeasible* on a feasible model (#1542). Every scalar affine subtree is
  therefore rebuilt as ``sum_i a_i z_i + k`` with the constants folded, so
  ``(z + c) - c`` becomes exactly ``z``. With folding the same experiment showed
  0 false certificates and 0 crashes (69 models).
* **Selectivity.** Only a variable whose |anchor| is at least ``ratio`` times its
  box width (and at least ``ratio``) is moved: recentring a well-scaled model is
  not free (st_e36: 89 -> 93 nodes, 11 -> 19 s), and a model with nothing to move
  is solved exactly as before.

Gated by ``DISCOPT_RECENTRE`` (default off, CLAUDE.md §5); the threshold by
``DISCOPT_RECENTRE_RATIO`` (default 100).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

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

logger = logging.getLogger(__name__)

FLAG = "DISCOPT_RECENTRE"
RATIO_ENV = "DISCOPT_RECENTRE_RATIO"
DEFAULT_RATIO = 100.0


class RecentreUnsupported(ValueError):
    """The model has structure this pass does not rebuild; solve it unrecentred."""


def recentre_enabled() -> bool:
    """``DISCOPT_RECENTRE`` -- default OFF until a graduation panel passes."""
    return os.environ.get(FLAG, "0") != "0"


def recentre_ratio() -> float:
    raw = os.environ.get(RATIO_ENV)
    ratio = DEFAULT_RATIO if raw is None else float(raw)
    if not np.isfinite(ratio) or ratio <= 0.0:
        raise ValueError(f"{RATIO_ENV} must be a positive finite number, got {raw!r}")
    return ratio


#: A bound at or beyond this magnitude is "effectively open" for the one-sided
#: rule: ``[1e6, 1e15]`` and ``[1e6, inf)`` are both "x >= 1e6".
_OPEN = 1e10


def plan_shifts(model: Model, ratio: float) -> dict[int, np.ndarray]:
    """``{id(var): c}`` for every variable worth moving, ``c`` shaped like the var.

    A component moves when either

    * **narrow offset** -- both bounds finite and ``|lb| >= ratio * max(1, ub-lb)``
      (anchor ``lb``); or
    * **one-sided offset** -- a finite ``lb`` with ``|lb| >= ratio`` on a box at
      least as wide as that offset (an open ``ub`` counts as infinitely wide):
      anchor ``lb``; or an open ``lb`` with ``|ub| >= ratio``: anchor ``ub``.
      Without this, ``x in [1e6, 1e15]`` keeps a user's ``x - 1e6`` unfolded and
      st_miqp4 / alan still certify infeasible / crash under a shift (#1543).

    Measured on the unshifted in-repo corpus at ratio 100: 4 of 66 models have a
    component to move (ex14_1_9 1, hda 2, heatexch_gen2 13, tanksize 12); the
    other 62 are left exactly as they were.

    The shift is the anchor rounded to an integer, so integer and binary domains
    stay integral.
    """
    plan: dict[int, np.ndarray] = {}
    for v in model._variables:
        lb = np.asarray(v.lb, dtype=np.float64)
        ub = np.asarray(v.ub, dtype=np.float64)
        lb_fin = np.isfinite(lb) & (np.abs(lb) < _OPEN)
        ub_fin = np.isfinite(ub) & (np.abs(ub) < _OPEN)
        width = np.where(lb_fin & ub_fin, ub - lb, np.inf)
        narrow = lb_fin & ub_fin & (np.abs(lb) >= ratio * np.maximum(1.0, width))
        # One-sided: the box is at least as wide as its offset (an open bound gives
        # width = inf). A fixed "open" cutoff alone missed x in [-1.3e6, 9.9987e9]
        # (a 1e10 bound after a shift) and left its (y - c)**2 unfolded -- the #1543
        # trigger, certified 262 vs the true 2 on shifted st_miqp2.
        from_lb = lb_fin & ~narrow & (np.abs(lb) >= ratio) & (width >= np.abs(lb))
        from_ub = ub_fin & ~lb_fin & (np.abs(ub) >= ratio)
        if np.any(narrow | from_lb | from_ub):
            plan[id(v)] = np.where(
                narrow | from_lb, np.round(lb), np.where(from_ub, np.round(ub), 0.0)
            )
    return plan


# ── affine canonicalisation ───────────────────────────────────────────────────

# An affine form: ({leaf_key: coef}, const, {leaf_key: leaf_node}).
_Aff = tuple[dict, float, dict]


def _scalar_index_key(node: IndexExpression) -> Optional[tuple]:
    base = node.base
    if not isinstance(base, Variable):
        return None
    idx = node.index
    if isinstance(idx, (int, np.integer)):
        idx = (int(idx),)
    if not (isinstance(idx, tuple) and all(isinstance(i, (int, np.integer)) for i in idx)):
        return None
    if len(idx) != len(base.shape):
        return None
    return (id(base), int(np.ravel_multi_index(tuple(int(i) for i in idx), base.shape)))


def _const_value(node) -> Optional[float]:
    if isinstance(node, Constant):
        val = np.asarray(node.value)
        if val.ndim == 0:
            return float(val)
    return None


def _canonical(aff: _Aff) -> Expression:
    coefs, const, leaves = aff
    terms: list[Expression] = []
    for key, a in coefs.items():
        if a == 0.0:
            continue
        leaf = leaves[key]
        terms.append(leaf if a == 1.0 else Constant(a) * leaf)
    if const != 0.0 or not terms:
        terms.append(Constant(const))
    return terms[0] if len(terms) == 1 else SumOverExpression(terms)


class _Rewriter:
    """``x -> z + c`` substitution with every scalar affine subtree folded."""

    def __init__(self, var_map: dict[int, Variable], shifts: dict[int, np.ndarray]):
        self.var_map = var_map  # id(old var) -> new var
        self.shifts = shifts  # id(old var) -> c (only moved vars)
        self.memo: dict[int, tuple[Expression, Optional[_Aff]]] = {}

    def __call__(self, expr: Expression) -> Expression:
        node, aff = self._rw(expr)
        return _canonical(aff) if aff is not None and not _is_leaf(node) else node

    def _rw(self, expr: Expression) -> tuple[Expression, Optional[_Aff]]:
        key = id(expr)
        hit = self.memo.get(key)
        if hit is None:
            hit = self._rw_impl(expr)
            self.memo[key] = hit
        return hit

    def _leaf_aff(self, node: Expression, leaf_key: tuple, shift: float) -> _Aff:
        return ({leaf_key: 1.0}, float(shift), {leaf_key: node})

    def _rw_impl(self, expr: Expression) -> tuple[Expression, Optional[_Aff]]:
        if isinstance(expr, Constant):
            v = _const_value(expr)
            return expr, (None if v is None else ({}, v, {}))
        if isinstance(expr, Parameter):
            return expr, None  # mutable: never folded into a constant
        if isinstance(expr, Variable):
            z = self.var_map[id(expr)]
            c = self.shifts.get(id(expr))
            if z.shape:
                node = z if c is None else z + Constant(c)
                return node, None  # array-valued: rewritten, not folded
            shift = 0.0 if c is None else float(np.asarray(c))
            node = z if shift == 0.0 else z + Constant(shift)
            return node, self._leaf_aff(z, (id(z), None), shift)
        if isinstance(expr, IndexExpression):
            if isinstance(expr.base, Variable):
                z = self.var_map[id(expr.base)]
                c = self.shifts.get(id(expr.base))
                leaf = IndexExpression(z, expr.index)
                ck = c[expr.index] if c is not None else None
                lkey = _scalar_index_key(leaf)
                if lkey is not None:
                    shift = 0.0 if ck is None else float(np.asarray(ck))
                    node = leaf if shift == 0.0 else leaf + Constant(shift)
                    return node, self._leaf_aff(leaf, lkey, shift)
                if ck is None or not np.any(ck):
                    return leaf, None
                return leaf + Constant(np.asarray(ck, dtype=np.float64)), None
            return IndexExpression(self(expr.base), expr.index), None
        if isinstance(expr, UnaryOp):
            child, caff = self._rw(expr.operand)
            node = UnaryOp(expr.op, self._emit(child, caff))
            if expr.op == "neg" and caff is not None:
                coefs, const, leaves = caff
                return node, ({k: -a for k, a in coefs.items()}, -const, leaves)
            return node, None
        if isinstance(expr, BinaryOp):
            left, laff = self._rw(expr.left)
            right, raff = self._rw(expr.right)
            node = BinaryOp(expr.op, self._emit(left, laff), self._emit(right, raff))
            return node, _combine(expr.op, laff, raff)
        if type(expr) is FunctionCall:
            return FunctionCall(expr.func_name, *(self(a) for a in expr.args)), None
        if isinstance(expr, MatMulExpression):
            return MatMulExpression(self(expr.left), self(expr.right)), None
        if isinstance(expr, SumExpression):
            return SumExpression(self(expr.operand), axis=expr.axis), None
        if isinstance(expr, SumOverExpression):
            parts = [self._rw(t) for t in expr.terms]
            node = SumOverExpression([self._emit(n, a) for n, a in parts])
            if all(a is not None for _, a in parts):
                acc: _Aff = ({}, 0.0, {})
                for _, a in parts:
                    acc = _combine("+", acc, a)  # type: ignore[assignment]
                return node, acc
            return node, None
        raise RecentreUnsupported(f"no rewrite rule for {type(expr).__name__}")

    @staticmethod
    def _emit(node: Expression, aff: Optional[_Aff]) -> Expression:
        """A child as it appears under a NON-affine parent: folded if affine."""
        return _canonical(aff) if aff is not None and not _is_leaf(node) else node


def _is_leaf(node: Expression) -> bool:
    return isinstance(node, (Constant, Parameter, Variable)) or (
        isinstance(node, IndexExpression) and isinstance(node.base, Variable)
    )


def _combine(op: str, a: Optional[_Aff], b: Optional[_Aff]) -> Optional[_Aff]:
    if a is None or b is None:
        return None
    (ca, ka, la), (cb, kb, lb) = a, b
    if op in ("+", "-"):
        sign = 1.0 if op == "+" else -1.0
        coefs = dict(ca)
        for k, v in cb.items():
            coefs[k] = coefs.get(k, 0.0) + sign * v
        return coefs, ka + sign * kb, {**la, **lb}
    if op == "*":
        if not ca:  # constant * affine
            return {k: ka * v for k, v in cb.items()}, ka * kb, lb
        if not cb:
            return {k: kb * v for k, v in ca.items()}, ka * kb, la
        return None
    if op == "/" and not cb and kb != 0.0:
        return {k: v / kb for k, v in ca.items()}, ka / kb, la
    return None


# ── the rebuilt model ─────────────────────────────────────────────────────────


@dataclass
class Recentring:
    """A recentred model plus the per-variable shifts to map points across."""

    model: Model
    shifts: dict[str, np.ndarray]  # var name -> c, for moved variables only

    def to_inner(self, values: dict[str, Any]) -> dict[str, Any]:
        """A point keyed by variable NAME, original coordinates -> recentred."""
        out = {}
        for name, val in values.items():
            c = self.shifts.get(name)
            out[name] = val if c is None else np.asarray(val, dtype=np.float64) - c
        return out

    def to_outer(self, values: dict[str, Any]) -> dict[str, Any]:
        out = {}
        for name, val in values.items():
            c = self.shifts.get(name)
            out[name] = val if c is None else np.asarray(val, dtype=np.float64) + c
        return out


def recentre(model: Model, ratio: Optional[float] = None) -> Optional[Recentring]:
    """The recentred twin of ``model``, or ``None`` when no variable qualifies.

    Raises :class:`RecentreUnsupported` for structure it does not rebuild
    (indicator / disjunctive / SOS / logical rows, piecewise domains,
    complementarities, a missing objective); the caller then solves unrecentred.
    """
    plan = plan_shifts(model, recentre_ratio() if ratio is None else ratio)
    if not plan:
        return None
    if model._objective is None:
        raise RecentreUnsupported("model has no objective")
    for con in model._constraints:
        if type(con) is not Constraint:
            raise RecentreUnsupported(f"a {type(con).__name__} row")
    for attr in ("_piecewise_domains", "_complementarities", "_lowered_complementarities"):
        if getattr(model, attr, None):
            raise RecentreUnsupported(f"model carries {attr}")

    new = Model(f"{model.name}")
    var_map: dict[int, Variable] = {}
    shifts_by_id: dict[int, np.ndarray] = {}
    shifts_by_name: dict[str, np.ndarray] = {}
    for v in model._variables:
        c = plan.get(id(v))
        lb, ub = np.asarray(v.lb, dtype=np.float64), np.asarray(v.ub, dtype=np.float64)
        if c is not None:
            c = np.broadcast_to(c, lb.shape).astype(np.float64)
            lb, ub = lb - c, ub - c
            shifts_by_id[id(v)] = c if v.shape else np.asarray(float(c))
            shifts_by_name[v.name] = shifts_by_id[id(v)]
        shape = v.shape if v.shape else ()
        if v.var_type is VarType.CONTINUOUS:
            z = new.continuous(v.name, shape=shape, lb=lb, ub=ub)
        elif v.var_type is VarType.BINARY and c is None:
            z = new.binary(v.name, shape=shape)
        else:
            z = new.integer(v.name, shape=shape, lb=lb, ub=ub)
        var_map[id(v)] = z
    for p in model._parameters:  # shared objects: values stay live, names unchanged
        new._parameters.append(p)
        new._names.add(p.name)

    rw = _Rewriter(var_map, shifts_by_id)
    for con in model._constraints:
        new._constraints.append(
            Constraint(body=rw(con.body), sense=con.sense, rhs=0.0, name=con.name)
        )
    new._objective = Objective(
        expression=rw(model._objective.expression), sense=model._objective.sense
    )
    # ``_initial_point`` is keyed ``(var name, flat element) -> value`` (set by
    # ``set_initial_point`` / ``from_nl``); move each entry into the new coordinates.
    new._initial_point = {
        (name, elem): float(val) - float(np.ravel(shifts_by_name[name])[elem])
        if name in shifts_by_name
        else val
        for (name, elem), val in getattr(model, "_initial_point", {}).items()
    }
    return Recentring(model=new, shifts=shifts_by_name)


def solve_recentred(outer: Model, rc: Recentring, solve_args: dict[str, Any]):
    """Solve ``rc.model`` with ``outer.solve``'s arguments; map the result back.

    The inner solve runs the whole ``Model.solve`` pipeline (every route, the
    convex kernel included). The mapped point is then re-verified against the
    ORIGINAL model: a certificate is only kept for a point the user's model
    accepts, so a mapping defect can never publish one.
    """
    import dataclasses

    from discopt.validation.feasibility import verify_point

    args = dict(solve_args)
    init = args.get("initial_solution")
    if init:
        by_name = {getattr(k, "name", k): v for k, v in init.items()}
        inner_vars = {v.name: v for v in rc.model._variables}
        args["initial_solution"] = {inner_vars[n]: val for n, val in rc.to_inner(by_name).items()}
    ws = args.get("warm_start")
    if ws is not None and ws.x is not None:
        args["warm_start"] = dataclasses.replace(ws, x=rc.to_inner(ws.x))

    rc.model._recentre_inner = True
    result = rc.model.solve(**args)

    if result.x is not None:
        result.x = rc.to_outer(result.x)
    result._model = outer
    if result.infeasibility_certificate is not None:
        # A row-violation witness of the recentred LP; its rows are the same, but it
        # was computed in z-space, so it is not handed out as the original's.
        result.infeasibility_certificate = None
    stats = result.solver_stats if result.solver_stats is not None else {}
    result.solver_stats = stats
    stats["recentre/variables_moved"] = float(len(rc.shifts))

    if result.x is not None and result.gap_certified and result.status == "optimal":
        flat = np.concatenate(
            [np.ravel(np.asarray(result.x[v.name], dtype=np.float64)) for v in outer._variables]
        )
        check = verify_point(outer, flat)
        if not check.ok:
            logger.warning(
                "recentring: the mapped incumbent fails the original model (%s); "
                "certificate withdrawn",
                check.reason,
            )
            result.gap_certified = False
            result.status = "feasible"
            stats["recentre/mapped_point_refused"] = 1.0
    return result
