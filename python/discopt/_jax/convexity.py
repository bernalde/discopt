"""
Convexity Detection for Expression DAGs.

Walks the expression DAG and classifies each (sub)expression as one of:
  CONVEX, CONCAVE, AFFINE, or UNKNOWN

using standard composition rules from convex analysis. Results are cached
on the expression objects.

Key composition rules implemented:
  - Constant/Variable/Parameter: AFFINE
  - sum/neg of convex  : CONVEX; sum/neg of concave : CONCAVE
  - const * expr       : preserves curvature if const >= 0, flips if < 0
  - convex(affine)     : CONVEX  (composition rule)
  - concave(affine)    : CONCAVE
  - exp(convex)        : CONVEX  (exp is convex and nondecreasing on the reals)
  - log(concave)       : CONCAVE (log is concave and nondecreasing)
  - x**2               : CONVEX  (for any x)
  - x**p (p>=1, x>=0)  : CONVEX  (on the nonneg reals)
  - bilinear x*y       : UNKNOWN (nonconvex in general)
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np

from discopt._jax.problem_classifier import (
    _extract_linear_coefficients,
    _extract_quadratic_coefficients,
)
from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Constraint,
    Expression,
    FunctionCall,
    IndexExpression,
    MatMulExpression,
    Model,
    Parameter,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
)


class Curvature(Enum):
    """Curvature classification of an expression."""

    AFFINE = "affine"
    CONVEX = "convex"
    CONCAVE = "concave"
    UNKNOWN = "unknown"


def _combine_sum(a: Curvature, b: Curvature) -> Curvature:
    """Curvature of a + b."""
    if a == Curvature.AFFINE:
        return b
    if b == Curvature.AFFINE:
        return a
    if a == b:
        return a  # convex + convex = convex, etc.
    return Curvature.UNKNOWN


def _negate(c: Curvature) -> Curvature:
    """Curvature of -expr given curvature of expr."""
    if c == Curvature.CONVEX:
        return Curvature.CONCAVE
    if c == Curvature.CONCAVE:
        return Curvature.CONVEX
    return c  # AFFINE stays AFFINE, UNKNOWN stays UNKNOWN


def _scale(c: Curvature, sign: int) -> Curvature:
    """Curvature of k * expr where sign = sign(k).

    sign: +1, -1, or 0 (constant zero).
    """
    if sign == 0 or c == Curvature.AFFINE:
        return Curvature.AFFINE
    if sign > 0:
        return c
    return _negate(c)


def _is_nonneg_domain(expr: Expression, model: Model) -> bool:
    """Check if expression always evaluates to nonneg values (lb >= 0)."""
    if isinstance(expr, Constant):
        return bool(np.all(expr.value >= 0))
    if isinstance(expr, Variable):
        return bool(np.all(expr.lb >= 0))
    if isinstance(expr, IndexExpression):
        if isinstance(expr.base, Variable):
            idx = expr.index
            if isinstance(idx, int):
                return bool(expr.base.lb.flat[idx] >= 0)
            if isinstance(idx, tuple) and len(idx) == 1:
                return bool(expr.base.lb.flat[idx[0]] >= 0)
        return False
    if isinstance(expr, Parameter):
        return bool(np.all(expr.value >= 0))
    return False


def _has_positive_lower_bound(expr: Expression, model: Model) -> bool:
    """Check if an expression has a strictly positive lower bound in simple cases."""
    if isinstance(expr, Constant):
        return bool(np.all(expr.value > 0))
    if isinstance(expr, Variable):
        return bool(np.all(expr.lb > 0))
    if isinstance(expr, IndexExpression):
        if isinstance(expr.base, Variable):
            idx = expr.index
            if isinstance(idx, int):
                return bool(expr.base.lb.flat[idx] > 0)
            if isinstance(idx, tuple) and len(idx) == 1:
                return bool(expr.base.lb.flat[idx[0]] > 0)
        return False
    if isinstance(expr, Parameter):
        return bool(np.all(expr.value > 0))
    return False


def _same_expr(lhs: Expression, rhs: Expression) -> bool:
    """Best-effort structural equality for small expression-pattern checks."""
    if lhs is rhs:
        return True
    if type(lhs) is not type(rhs):
        return False

    if isinstance(lhs, Constant):
        return bool(np.array_equal(lhs.value, rhs.value))
    if isinstance(lhs, Parameter):
        return lhs.name == rhs.name and bool(np.array_equal(lhs.value, rhs.value))
    if isinstance(lhs, Variable):
        return lhs.name == rhs.name
    if isinstance(lhs, IndexExpression):
        return lhs.index == rhs.index and _same_expr(lhs.base, rhs.base)
    if isinstance(lhs, UnaryOp):
        return lhs.op == rhs.op and _same_expr(lhs.operand, rhs.operand)
    if isinstance(lhs, BinaryOp):
        return (
            lhs.op == rhs.op
            and _same_expr(lhs.left, rhs.left)
            and _same_expr(lhs.right, rhs.right)
        )
    if isinstance(lhs, FunctionCall):
        return lhs.func_name == rhs.func_name and len(lhs.args) == len(rhs.args) and all(
            _same_expr(a, b) for a, b in zip(lhs.args, rhs.args)
        )
    if isinstance(lhs, SumExpression):
        return _same_expr(lhs.operand, rhs.operand)
    if isinstance(lhs, SumOverExpression):
        return len(lhs.terms) == len(rhs.terms) and all(
            _same_expr(a, b) for a, b in zip(lhs.terms, rhs.terms)
        )
    return False


def _total_scalar_variables(model: Model) -> int:
    return sum(int(v.size) for v in model._variables)


def _scalar_var_offset(model: Model, target: Variable) -> Optional[int]:
    offset = 0
    for var in model._variables:
        if var is target:
            return offset
        offset += int(var.size)
    return None


def _quadratic_curvature(expr: Expression, model: Model) -> Optional[Curvature]:
    """Return quadratic curvature when an expression is a scalar quadratic."""
    try:
        Q, _c, _const = _extract_quadratic_coefficients(expr, model, _total_scalar_variables(model))
    except Exception:
        return None

    Q = 0.5 * (Q + Q.T)
    if np.allclose(Q, 0.0, atol=1e-10):
        return Curvature.AFFINE

    eigvals = np.linalg.eigvalsh(Q)
    if float(np.min(eigvals)) >= -1e-10:
        return Curvature.CONVEX
    if float(np.max(eigvals)) <= 1e-10:
        return Curvature.CONCAVE
    return Curvature.UNKNOWN


def _quadratic_data(expr: Expression, model: Model):
    """Extract scalar quadratic data or return None."""
    try:
        Q, c, const = _extract_quadratic_coefficients(expr, model, _total_scalar_variables(model))
    except Exception:
        return None
    return 0.5 * (Q + Q.T), c, const


def _is_homogeneous_psd_quadratic(expr: Expression, model: Model) -> bool:
    """Check if expr is x'Qx with Q psd and no linear/constant terms."""
    data = _quadratic_data(expr, model)
    if data is None:
        return False
    Q, c, const = data
    if not np.allclose(c, 0.0, atol=1e-10):
        return False
    if abs(float(const)) > 1e-10:
        return False
    eigvals = np.linalg.eigvalsh(Q)
    return float(np.min(eigvals)) >= -1e-10


def _flatten_product(expr: Expression, out: list[Expression]) -> None:
    if isinstance(expr, BinaryOp) and expr.op == "*":
        _flatten_product(expr.left, out)
        _flatten_product(expr.right, out)
        return
    out.append(expr)


def _extract_power_factor(expr: Expression) -> Optional[tuple[Expression, float]]:
    if isinstance(expr, BinaryOp) and expr.op == "**":
        if isinstance(expr.right, (Constant, Parameter)):
            exponent = float(np.asarray(expr.right.value))
            return expr.left, exponent
        return None
    return expr, 1.0


def _flatten_sum_terms(expr: Expression, scale: float, out: list[tuple[float, Expression]]) -> None:
    if isinstance(expr, BinaryOp) and expr.op == "+":
        _flatten_sum_terms(expr.left, scale, out)
        _flatten_sum_terms(expr.right, scale, out)
        return
    if isinstance(expr, BinaryOp) and expr.op == "-":
        _flatten_sum_terms(expr.left, scale, out)
        _flatten_sum_terms(expr.right, -scale, out)
        return
    if isinstance(expr, UnaryOp) and expr.op == "neg":
        _flatten_sum_terms(expr.operand, -scale, out)
        return
    out.append((scale, expr))


def _contains_var(expr: Expression, target: Variable) -> bool:
    if isinstance(expr, Variable):
        return expr is target or expr.name == target.name
    if isinstance(expr, IndexExpression):
        return isinstance(expr.base, Variable) and (
            expr.base is target or expr.base.name == target.name
        )
    if isinstance(expr, BinaryOp):
        return _contains_var(expr.left, target) or _contains_var(expr.right, target)
    if isinstance(expr, UnaryOp):
        return _contains_var(expr.operand, target)
    if isinstance(expr, FunctionCall):
        return any(_contains_var(arg, target) for arg in expr.args)
    if isinstance(expr, SumExpression):
        return _contains_var(expr.operand, target)
    if isinstance(expr, SumOverExpression):
        return any(_contains_var(term, target) for term in expr.terms)
    return False


def _constant_expr(value: float) -> Constant:
    return Constant(np.array(float(value), dtype=np.float64))


def _add_expr(lhs: Optional[Expression], rhs: Expression) -> Expression:
    if lhs is None:
        return rhs
    return BinaryOp("+", lhs, rhs)


def _scale_expr(expr: Expression, scale: float) -> Expression:
    if abs(scale - 1.0) <= 1e-12:
        return expr
    if abs(scale + 1.0) <= 1e-12:
        return UnaryOp("neg", expr)
    return BinaryOp("*", _constant_expr(scale), expr)


def _extract_linear_factor(expr: Expression, target: Variable) -> Optional[Expression]:
    if isinstance(expr, Variable) and (expr is target or expr.name == target.name):
        return _constant_expr(1.0)
    if isinstance(expr, IndexExpression):
        if isinstance(expr.base, Variable) and (
            expr.base is target or expr.base.name == target.name
        ):
            return _constant_expr(1.0)
        return None
    if isinstance(expr, UnaryOp) and expr.op == "neg":
        inner = _extract_linear_factor(expr.operand, target)
        return None if inner is None else UnaryOp("neg", inner)
    if isinstance(expr, BinaryOp) and expr.op == "*":
        left_has = _contains_var(expr.left, target)
        right_has = _contains_var(expr.right, target)
        if left_has and right_has:
            return None
        if left_has:
            inner = _extract_linear_factor(expr.left, target)
            return None if inner is None else BinaryOp("*", inner, expr.right)
        if right_has:
            inner = _extract_linear_factor(expr.right, target)
            return None if inner is None else BinaryOp("*", expr.left, inner)
    return None


def _affine_range_1d(alpha: float, beta: float, lb: float, ub: float) -> tuple[float, float]:
    if alpha >= 0.0:
        lo = alpha * lb + beta if np.isfinite(lb) else (-np.inf if alpha > 0.0 else beta)
        hi = alpha * ub + beta if np.isfinite(ub) else (np.inf if alpha > 0.0 else beta)
    else:
        lo = alpha * ub + beta if np.isfinite(ub) else (-np.inf)
        hi = alpha * lb + beta if np.isfinite(lb) else (np.inf)
    return float(lo), float(hi)


def _classify_fractional_epigraph_constraint(
    constraint: Constraint,
    model: Optional[Model],
) -> Optional[bool]:
    """Detect scalar epigraph/hypograph constraints for quadratic-over-affine univariate forms."""
    if model is None or constraint.sense != "<=":
        return None

    scalar_targets = [v for v in model._variables if v.size == 1]
    if len(scalar_targets) != 2:
        return None

    n = _total_scalar_variables(model)
    for target in scalar_targets:
        terms: list[tuple[float, Expression]] = []
        _flatten_sum_terms(constraint.body, 1.0, terms)

        coeff_expr: Optional[Expression] = None
        remainder_expr: Optional[Expression] = None
        valid = True
        for scale, term in terms:
            factor = _extract_linear_factor(term, target)
            if factor is None:
                if _contains_var(term, target):
                    valid = False
                    break
                remainder_expr = _add_expr(remainder_expr, _scale_expr(term, scale))
                continue
            coeff_expr = _add_expr(coeff_expr, _scale_expr(factor, scale))

        if not valid or coeff_expr is None or remainder_expr is None:
            continue

        try:
            coeff_vec, coeff_const = _extract_linear_coefficients(coeff_expr, model, n)
        except Exception:
            continue

        nonzero_coeff = np.flatnonzero(np.abs(coeff_vec) > 1e-10)
        target_idx = _scalar_var_offset(model, target)
        if target_idx is None:
            continue
        if target_idx in nonzero_coeff:
            continue
        if len(nonzero_coeff) != 1:
            continue
        other_idx = int(nonzero_coeff[0])

        data = _quadratic_data(remainder_expr, model)
        if data is None:
            continue
        Q, c, const = data
        remainder_support = set(np.flatnonzero(np.abs(np.diag(Q)) > 1e-10))
        remainder_support |= set(np.flatnonzero(np.abs(c) > 1e-10))
        if remainder_support - {other_idx}:
            continue
        if np.any(np.abs(Q[np.arange(Q.shape[0]) != other_idx, :]) > 1e-10):
            continue
        if np.any(np.abs(Q[:, np.arange(Q.shape[0]) != other_idx]) > 1e-10):
            continue

        other_var = None
        running = 0
        for var in model._variables:
            if running == other_idx and var.size == 1:
                other_var = var
                break
            running += var.size
        if other_var is None:
            continue

        a = 0.5 * float(Q[other_idx, other_idx])
        b = float(c[other_idx])
        c0 = float(const)
        d = float(coeff_vec[other_idx])
        e = float(coeff_const)
        lb = float(other_var.lb)
        ub = float(other_var.ub)
        coeff_lo, coeff_hi = _affine_range_1d(d, e, lb, ub)

        curvature_numerator = a * e * e - b * d * e + c0 * d * d
        if coeff_hi < -1e-10:
            # coeff(x) * y + r(x) <= 0 with coeff < 0  =>  y >= r(x) / (-coeff(x))
            return curvature_numerator >= -1e-10
        if coeff_lo > 1e-10:
            # coeff(x) * y + r(x) <= 0 with coeff > 0  =>  y <= -r(x) / coeff(x)
            return curvature_numerator <= 1e-10

    return None


def _classify_product_special(
    expr: BinaryOp,
    model: Optional[Model],
    cache: dict,
) -> Optional[Curvature]:
    if model is None:
        return None

    # Perspective of exp: y * exp(x / y), y > 0.
    for scale_expr, exp_expr in ((expr.left, expr.right), (expr.right, expr.left)):
        if (
            classify_expr(scale_expr, model, cache) == Curvature.AFFINE
            and _has_positive_lower_bound(scale_expr, model)
            and isinstance(exp_expr, FunctionCall)
            and exp_expr.func_name == "exp"
            and len(exp_expr.args) == 1
        ):
            inner = exp_expr.args[0]
            if isinstance(inner, BinaryOp) and inner.op == "/":
                if (
                    _same_expr(scale_expr, inner.right)
                    and classify_expr(inner.left, model, cache) == Curvature.AFFINE
                    and classify_expr(inner.right, model, cache) == Curvature.AFFINE
                ):
                    return Curvature.CONVEX

    # Weighted geometric mean / power cone primitives:
    # prod_i x_i^a_i with x_i >= 0, 0 <= a_i <= 1, sum a_i = 1.
    factors: list[Expression] = []
    _flatten_product(expr, factors)
    if len(factors) < 2:
        return None

    parsed: list[tuple[Expression, float]] = []
    for factor in factors:
        extracted = _extract_power_factor(factor)
        if extracted is None:
            return None
        base, exponent = extracted
        if exponent < -1e-10 or exponent > 1.0 + 1e-10:
            return None
        if classify_expr(base, model, cache) != Curvature.AFFINE:
            return None
        if not _is_nonneg_domain(base, model):
            return None
        parsed.append((base, exponent))

    if abs(sum(exponent for _, exponent in parsed) - 1.0) <= 1e-10:
        return Curvature.CONCAVE
    return None


def _classify_division_special(
    expr: BinaryOp,
    model: Optional[Model],
    cache: dict,
) -> Optional[Curvature]:
    if model is None:
        return None
    if classify_expr(expr.right, model, cache) != Curvature.AFFINE:
        return None
    if not _has_positive_lower_bound(expr.right, model):
        return None
    if _is_homogeneous_psd_quadratic(expr.left, model):
        return Curvature.CONVEX
    return None


def _classify_function_special(
    expr: FunctionCall,
    model: Optional[Model],
    cache: dict,
) -> Optional[Curvature]:
    if model is None:
        return None
    if expr.func_name == "sqrt" and len(expr.args) == 1:
        # sqrt(x'Qx) is a norm when Q is psd and the quadratic is homogeneous.
        if _is_homogeneous_psd_quadratic(expr.args[0], model):
            return Curvature.CONVEX
        arg_curv = classify_expr(expr.args[0], model, cache)
        if arg_curv in (Curvature.AFFINE, Curvature.CONCAVE):
            return Curvature.CONCAVE
        return Curvature.UNKNOWN
    return None


def classify_expr(
    expr: Expression,
    model: Optional[Model] = None,
    _cache: Optional[dict] = None,
) -> Curvature:
    """Classify the curvature of an expression.

    Args:
        expr: Expression to classify.
        model: Model context (used for variable bound lookups).
        _cache: Internal memoization dict (keyed by expression id).

    Returns:
        Curvature enum value.
    """
    if _cache is None:
        _cache = {}

    eid = id(expr)
    if eid in _cache:
        return _cache[eid]  # type: ignore[no-any-return]

    result = _classify_impl(expr, model, _cache)
    if result == Curvature.UNKNOWN and model is not None:
        quad_curv = _quadratic_curvature(expr, model)
        if quad_curv is not None:
            result = quad_curv
    _cache[eid] = result
    return result


def _classify_impl(
    expr: Expression,
    model: Optional[Model],
    cache: dict,
) -> Curvature:
    """Internal recursive classification."""

    # --- Leaves ---
    if isinstance(expr, (Constant, Parameter)):
        return Curvature.AFFINE

    if isinstance(expr, Variable):
        return Curvature.AFFINE

    if isinstance(expr, IndexExpression):
        base_curv = classify_expr(expr.base, model, cache)
        return base_curv  # indexing preserves curvature

    # --- Unary ops ---
    if isinstance(expr, UnaryOp):
        child = classify_expr(expr.operand, model, cache)
        if expr.op == "neg":
            return _negate(child)
        if expr.op == "abs":
            # |x| is convex when x is affine
            if child == Curvature.AFFINE:
                return Curvature.CONVEX
            return Curvature.UNKNOWN
        return Curvature.UNKNOWN

    # --- Binary ops ---
    if isinstance(expr, BinaryOp):
        left = classify_expr(expr.left, model, cache)
        right = classify_expr(expr.right, model, cache)

        if expr.op == "+":
            return _combine_sum(left, right)

        if expr.op == "-":
            return _combine_sum(left, _negate(right))

        if expr.op == "*":
            # const * expr
            if isinstance(expr.left, (Constant, Parameter)):
                val = np.asarray(expr.left.value)
                if val.ndim == 0:
                    s = 1 if float(val) >= 0 else -1
                    return _scale(right, s)
            if isinstance(expr.right, (Constant, Parameter)):
                val = np.asarray(expr.right.value)
                if val.ndim == 0:
                    s = 1 if float(val) >= 0 else -1
                    return _scale(left, s)
            special = _classify_product_special(expr, model, cache)
            if special is not None:
                return special
            return Curvature.UNKNOWN

        if expr.op == "/":
            # expr / const
            if isinstance(expr.right, (Constant, Parameter)):
                val = np.asarray(expr.right.value)
                if val.ndim == 0 and abs(float(val)) > 1e-30:
                    s = 1 if float(val) > 0 else -1
                    return _scale(left, s)
            special = _classify_division_special(expr, model, cache)
            if special is not None:
                return special
            return Curvature.UNKNOWN

        if expr.op == "**":
            # x^n where n is constant
            if isinstance(expr.right, (Constant, Parameter)):
                n_val = np.asarray(expr.right.value)
                if n_val.ndim == 0:
                    n = float(n_val)
                    n_int = int(n)
                    base = classify_expr(expr.left, model, cache)

                    # x^1 = x — preserves curvature
                    if np.isclose(n, 1.0):
                        return base

                    # x^0 = constant
                    if np.isclose(n, 0.0):
                        return Curvature.AFFINE

                    # x^2: convex for all x (affine base) or
                    # convex of convex requires monotonicity — x^2 is
                    # not monotone, so only valid for affine base.
                    if np.isclose(n, 2.0):
                        if base == Curvature.AFFINE:
                            return Curvature.CONVEX
                        return Curvature.UNKNOWN

                    # Even integer power >=2: convex on all of R
                    # when composed with affine
                    if np.isclose(n, float(n_int)) and n_int % 2 == 0 and n_int >= 2:
                        if base == Curvature.AFFINE:
                            return Curvature.CONVEX
                        return Curvature.UNKNOWN

                    # Odd integer power >= 3: convex on [0, inf),
                    # concave on (-inf, 0] — only convex when
                    # base is affine and nonneg
                    if np.isclose(n, float(n_int)) and n_int % 2 == 1 and n_int >= 3:
                        if base == Curvature.AFFINE:
                            if model is not None and _is_nonneg_domain(expr.left, model):
                                return Curvature.CONVEX
                        return Curvature.UNKNOWN

                    # Fractional: 0 < n < 1 and nonneg domain: concave
                    if 0 < n < 1:
                        if base == Curvature.AFFINE:
                            if model is not None and _is_nonneg_domain(expr.left, model):
                                return Curvature.CONCAVE
                        return Curvature.UNKNOWN

                    # n > 1 (non-integer), nonneg domain: convex
                    if n > 1:
                        if base == Curvature.AFFINE:
                            if model is not None and _is_nonneg_domain(expr.left, model):
                                return Curvature.CONVEX
                        return Curvature.UNKNOWN

            return Curvature.UNKNOWN

        return Curvature.UNKNOWN

    # --- Function calls ---
    if isinstance(expr, FunctionCall):
        name = expr.func_name
        special = _classify_function_special(expr, model, cache)
        if special is not None:
            return special
        if len(expr.args) == 1:
            arg_curv = classify_expr(expr.args[0], model, cache)

            # exp(x): convex & nondecreasing
            # exp(convex) = convex, exp(affine) = convex
            if name == "exp":
                if arg_curv in (Curvature.AFFINE, Curvature.CONVEX):
                    return Curvature.CONVEX
                return Curvature.UNKNOWN

            # log(x): concave & nondecreasing
            # log(concave) = concave, log(affine) = concave
            if name in ("log", "log2", "log10"):
                if arg_curv in (Curvature.AFFINE, Curvature.CONCAVE):
                    return Curvature.CONCAVE
                return Curvature.UNKNOWN

            # sqrt(x): concave & nondecreasing on nonneg
            if name == "sqrt":
                return Curvature.UNKNOWN

            # abs(x): convex (nondecreasing for x>0, nonincreasing for x<0
            # — satisfies DCP when composed with affine)
            if name == "abs":
                if arg_curv == Curvature.AFFINE:
                    return Curvature.CONVEX
                return Curvature.UNKNOWN

            # cosh(x): convex & even function
            if name == "cosh":
                if arg_curv == Curvature.AFFINE:
                    return Curvature.CONVEX
                return Curvature.UNKNOWN

            # sin, cos, tan, sinh, tanh, asin, acos, atan:
            # not globally convex or concave
            return Curvature.UNKNOWN

        # Multi-arg functions: unknown
        return Curvature.UNKNOWN

    # --- Sum expressions ---
    if isinstance(expr, SumExpression):
        child = classify_expr(expr.operand, model, cache)
        return child  # sum preserves curvature

    if isinstance(expr, SumOverExpression):
        result = Curvature.AFFINE
        for t in expr.terms:
            t_curv = classify_expr(t, model, cache)
            result = _combine_sum(result, t_curv)
            if result == Curvature.UNKNOWN:
                return Curvature.UNKNOWN
        return result

    # --- MatMul ---
    if isinstance(expr, MatMulExpression):
        # A @ x where A is constant: affine
        left = classify_expr(expr.left, model, cache)
        right = classify_expr(expr.right, model, cache)
        if isinstance(expr.left, (Constant, Parameter)):
            return right  # const @ expr preserves curvature
        if isinstance(expr.right, (Constant, Parameter)):
            return left  # expr @ const preserves curvature
        return Curvature.UNKNOWN

    return Curvature.UNKNOWN


def classify_constraint(
    constraint: Constraint,
    model: Optional[Model] = None,
    _cache: Optional[dict] = None,
) -> bool:
    """Check if a constraint is convex.

    A constraint g(x) <= 0 is convex when g is convex.
    A constraint g(x) >= 0 is convex when g is concave (i.e., -g convex).
    A constraint g(x) == 0 is convex only when g is affine.

    Returns:
        True if the constraint is convex, False otherwise.
    """
    if _cache is None:
        _cache = {}

    curv = classify_expr(constraint.body, model, _cache)

    if constraint.sense == "<=":
        # body <= rhs means body - rhs <= 0
        # Convex if body is convex
        if curv in (Curvature.CONVEX, Curvature.AFFINE):
            return True
        special = _classify_fractional_epigraph_constraint(constraint, model)
        if special is not None:
            return special
        return False
    elif constraint.sense == ">=":
        # body >= rhs means rhs - body <= 0
        # Convex if body is concave (so -body is convex)
        return curv in (Curvature.CONCAVE, Curvature.AFFINE)
    elif constraint.sense == "==":
        # Equality: convex only if affine
        return curv == Curvature.AFFINE
    return False


def classify_model(model: Model) -> tuple[bool, list[bool]]:
    """Classify a model's convexity.

    Returns:
        (is_convex, constraint_convexity_mask)
        - is_convex: True if objective is convex and all constraints are convex
        - constraint_convexity_mask: per-constraint convexity flags
    """
    cache: dict = {}

    # Check objective
    obj_convex = True
    if model._objective is not None:
        from discopt.modeling.core import ObjectiveSense

        obj_curv = classify_expr(model._objective.expression, model, cache)
        if model._objective.sense == ObjectiveSense.MINIMIZE:
            obj_convex = obj_curv in (Curvature.CONVEX, Curvature.AFFINE)
        else:
            # Maximize f  ≡ Minimize -f; convex if f is concave
            obj_convex = obj_curv in (Curvature.CONCAVE, Curvature.AFFINE)

    # Check constraints
    constraint_mask = []
    all_convex = obj_convex
    for c in model._constraints:
        if isinstance(c, Constraint):
            is_cvx = classify_constraint(c, model, cache)
            constraint_mask.append(is_cvx)
            if not is_cvx:
                all_convex = False
        else:
            constraint_mask.append(False)
            all_convex = False

    return all_convex, constraint_mask
