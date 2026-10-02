"""Regression tests for #1542: false certificates from cancellation at large offsets.

Two independent routes, each fixed at its source:

* **A - the separable objective floor.** ``_expression_lower_bound_for_lift``
  distributed ``0.2*(y - c)**3`` in floating point into monomials of magnitude
  ``~c**3`` (``1.9e18`` at ``c = 1228561``; one ulp is 256) whose exact sum is
  ``O(1)``, and evaluated them in that raw basis. The rounded coefficients were
  wrong by more than the objective's range, the floor came out at ``+59`` on a
  model whose optimum is ``-45``, and the ``obj >= floor`` row cut the optimum
  off: the root closed at a false optimum. Univariate polynomial terms are now
  built from the expression's own structure in exact rational arithmetic and
  bounded about an anchor inside the box with exact evaluation.

* **B - affine constants carried through distribution.** ``(z + c) - c`` is
  ``z``, but ``distribute_products`` multiplied the cancelling constants into
  every product, so the factorable reform's ``_fr_aux_*`` rows carried
  ``1e12``-scale terms whose float cancellation made st_e36's root McCormick LP
  infeasible: a feasible model certified infeasible. Each maximal affine subtree
  is now folded exactly before distribution.
"""

from __future__ import annotations

import os
from fractions import Fraction

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.milp_relaxation import (
    _exact_polynomial_lower_bound,
    _exact_univariate_polynomial,
    _expression_lower_bound_for_lift,
)
from discopt._relax.term_classifier import distribute_products, fold_affine_constants
from discopt.modeling.core import BinaryOp, Constant, from_nl
from discopt.validation.feasibility import verify_point

DATA = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl")


def _same_certificate(base, other):
    """Both certified optimal at the same objective (#1537's tolerance)."""
    assert base.gap_certified and base.status == "optimal", base
    assert other.gap_certified, "the shifted solve lost its certificate"
    assert other.status == "optimal", f"certified {other.status} on a feasible model"
    tol = max(1e-6, 1e-4 * max(1.0, abs(base.objective)))
    assert abs(other.objective - base.objective) <= tol, (
        f"certified {other.objective!r}, unshifted model certified {base.objective!r}"
    )


# ── route A: whole-solve reproductions ───────────────────────────────────────


def _cube(shift: float) -> dm.Model:
    """``min x**3 - 2x`` on ``x in [-2, 3]``, written in ``y = x + shift``."""
    m = dm.Model("cube")
    y = m.continuous("y", lb=-2 + shift, ub=3 + shift)
    x = y - Constant(shift)
    m.minimize(x**3 - 2 * x)
    return m


@pytest.mark.correctness
@pytest.mark.parametrize("shift", [1228561.0, -1314226.0])
def test_shifted_cubic_certifies_true_optimum(shift):
    # On main the +1228561 case certified -1.0887 (true optimum -4, at x = -2).
    base = _cube(0.0).solve(time_limit=30)
    _same_certificate(base, _cube(shift).solve(time_limit=30))


#: ``translate(polynomial(seed), 1e6, seed=2)`` of the #1537 harness: its draw.
_POLY_SHIFT = np.array([798491.0, -1314226.0, -1100101.0, -1228561.0, 555147.0])


def _polynomial(seed: int, shift: np.ndarray) -> dm.Model:
    """The #1537 harness's ``polynomial(seed)`` family, written in ``y = x + shift``."""
    rng = np.random.default_rng(seed)
    m = dm.Model(f"poly{seed}")
    x = []
    for k in range(5):
        lo, hi, integer = (0, 4, True) if k < 3 else (0, 5, False)
        name = f"i{k}" if integer else f"c{k - 3}"
        make = m.integer if integer else m.continuous
        y = make(name, lb=lo + shift[k], ub=hi + shift[k])
        x.append(y - Constant(float(shift[k])) if shift[k] else y)
    A = rng.integers(-4, 5, size=(3, 5)).astype(float)
    b = rng.integers(2, 10, size=3).astype(float)
    for r in range(3):
        m.subject_to(sum(A[r, j] * x[j] for j in range(5)) <= b[r])
    c = rng.integers(-5, 6, size=5).astype(float)
    m.minimize(sum(c[j] * x[j] for j in range(5)) + 0.2 * x[3] ** 3 - x[4] ** 2)
    return m


@pytest.mark.correctness
@pytest.mark.parametrize("seed", [1, 3])
def test_shifted_polynomial_family_certifies_true_optimum(seed):
    # On main: seed 1 certified -36.99999999 (true -45.0), seed 3 certified
    # -43.86 (true -45.86), both closed at the root by the false objective floor.
    base = _polynomial(seed, np.zeros(5)).solve(time_limit=30)
    _same_certificate(base, _polynomial(seed, _POLY_SHIFT).solve(time_limit=30))


# ── route A: the bound itself (differential test on fixed boxes) ─────────────


def _true_cubic_min(a3, a2, a1, lo, hi):
    """Min of ``a3 x^3 + a2 x^2 + a1 x`` on ``[lo, hi]`` (well-conditioned: |x| small)."""
    pts = [lo, hi]
    for r in np.roots([3 * a3, 2 * a2, a1]):
        if abs(r.imag) < 1e-12 and lo <= r.real <= hi:
            pts.append(r.real)
    return min(a3 * p**3 + a2 * p**2 + a1 * p for p in pts)


@pytest.mark.correctness
def test_separable_floor_is_valid_and_tight_under_large_shifts():
    rng = np.random.default_rng(1542)
    checked = 0
    for _ in range(60):
        a3, a2, a1 = rng.uniform(-1, 1), rng.uniform(-2, 2), rng.uniform(-5, 5)
        lo = float(rng.uniform(-3, 2))
        hi = lo + float(rng.uniform(0.5, 4))
        truth = _true_cubic_min(a3, a2, a1, lo, hi)
        for shift in (0.0, 987.0, 1228561.0, -1314226.0, 3.0e7):
            m = dm.Model("floor")
            y = m.continuous("y", lb=lo + shift, ub=hi + shift)
            x = y - Constant(shift)
            expr = a3 * x**3 + a2 * x**2 + a1 * x
            m.minimize(expr)
            bound = _expression_lower_bound_for_lift(
                expr, m, np.array([lo + shift]), np.array([hi + shift])
            )
            assert bound is not None
            # Valid: never above the true minimum (the box ends are exact floats
            # here only up to the shift's rounding, hence the 1e-7 slack).
            assert bound <= truth + 1e-7, (shift, bound, truth)
            # Tight: the exact path loses nothing to the shift.
            assert bound >= truth - 1e-6, (shift, bound, truth)
            checked += 1
    assert checked == 300


@pytest.mark.correctness
def test_exact_polynomial_is_exact():
    m = dm.Model("p")
    y = m.continuous("y", lb=0, ub=1)
    c = 1228561.0
    var_idx, coeffs = _exact_univariate_polynomial(Constant(0.2) * (y - Constant(c)) ** 3, m)
    assert var_idx == 0
    k = Fraction(0.2)
    C = Fraction(c)
    assert coeffs == {3: k, 2: -3 * k * C, 1: 3 * k * C**2, 0: -k * C**3}
    # Two variables, or division by a variable, are not univariate polynomials.
    z = m.continuous("z", lb=0, ub=1)
    assert _exact_univariate_polynomial(y * z, m) is None
    assert _exact_univariate_polynomial(Constant(1.0) / y, m) is None
    # Exact zero coefficients cancel; tiny ones are kept (the old path dropped
    # |coeff| <= 1e-12, which is unsound once |x| is large).
    assert _exact_polynomial_lower_bound({1: Fraction(1e-13)}, 1e15, 2e15) == Fraction(
        1e-13
    ) * Fraction(1e15)


@pytest.mark.correctness
def test_polynomial_floor_never_exceeds_true_minimum():
    """The floor is a rigorous bound, not the value at a float critical point.

    q evaluated exactly at np.roots' approximate minimizer over-estimates the
    minimum (by ~1e-31 here, on 21 of these 300 cases before the Bernstein
    branch-and-bound); compare against sympy's exact algebraic minimum.
    """
    sp = pytest.importorskip("sympy")
    import random

    u = sp.symbols("u")
    rng = random.Random(0)
    compared = 0
    for _ in range(300):
        deg = rng.choice([2, 3, 4, 5, 6])
        coeffs = {k: Fraction(rng.randint(-9, 9), rng.randint(1, 7)) for k in range(deg + 1)}
        if deg % 2 == 0:
            coeffs[deg] = abs(coeffs[deg]) + 1
        elif coeffs[deg] == 0:
            coeffs[deg] = Fraction(1)
        shift = rng.choice([0, 987, 1228561, -1314226])
        lo, hi = sorted([rng.uniform(-3, 3), rng.uniform(-3, 3)])
        lo, hi = lo + shift, hi + shift
        bound = _exact_polynomial_lower_bound(coeffs, lo, hi)
        q = sum(sp.Rational(c.numerator, c.denominator) * u**k for k, c in coeffs.items())
        L, H = (
            sp.Rational(*Fraction(lo).as_integer_ratio()),
            sp.Rational(*Fraction(hi).as_integer_ratio()),
        )
        crit = [r for r in sp.real_roots(sp.Poly(sp.diff(q, u), u)) if L <= r <= H]
        b = sp.Rational(bound.numerator, bound.denominator)
        for r in [L, H, *crit]:
            diff = q.subs(u, r) - b
            ok = diff >= 0 if diff.is_Rational else sp.N(diff, 400) >= 0
            assert ok, (coeffs, lo, hi, sp.N(diff, 30))
        true_min = min(sp.N(q.subs(u, r), 60) for r in [L, H, *crit])
        assert b >= true_min - sp.Rational(1, 10**6) * max(1, abs(true_min))
        compared += 1
    assert compared == 300


# ── route B: affine folding ahead of distribution ────────────────────────────


@pytest.mark.correctness
def test_fold_affine_constants_cancels_offsets_exactly():
    m = dm.Model("f")
    z = m.continuous("z", lb=0, ub=1)
    w = m.continuous("w", lb=0, ub=1)
    c = Constant(1228561.3)
    assert fold_affine_constants((z + c) - c) is z
    folded = fold_affine_constants(((z + c) - c) * w)
    assert isinstance(folded, BinaryOp) and folded.op == "*"
    assert folded.left is z and folded.right is w
    # Distribution no longer manufactures cancelling c*w terms.
    assert repr(distribute_products(((z + c) - c) * w)) == repr(z * w)


@pytest.mark.correctness
def test_fold_affine_constants_preserves_identity_when_nothing_combines():
    m = dm.Model("f")
    z = m.continuous("z", lb=0, ub=1)
    w = m.continuous("w", lb=0, ub=1)
    expr = 2 * (z + 3) * w + dm.exp(z - 1)
    assert fold_affine_constants(expr) is expr
    # A protected node is never rebuilt, even when it would fold.
    cancel = (z + Constant(5.0)) - Constant(5.0)
    outer = cancel * w
    folded = fold_affine_constants(outer, frozenset({id(cancel)}))
    assert folded is outer
    # A FunctionCall keeps its identity (id()-keyed univariate lift maps).
    call = dm.exp((z + Constant(5.0)) - Constant(5.0))
    assert fold_affine_constants(call + w) is not None
    assert fold_affine_constants(call) is call


def _cancel_shift(model, cs):
    """``model`` with each scalar variable ``x`` written ``(z + c) - c``, z on x's box."""
    from discopt.modeling.core import Constraint, Model, Objective, VarType

    new = Model(model.name + "_cancel")
    vmap = {}
    for v, c in zip(model._variables, cs):
        make = new.continuous if v.var_type is VarType.CONTINUOUS else new.integer
        z = make(v.name, lb=float(v.lb), ub=float(v.ub))
        vmap[id(v)] = (z + Constant(c)) - Constant(c)

    def sub(e):
        from discopt.modeling.core import FunctionCall, IndexExpression, UnaryOp, Variable

        if isinstance(e, Variable):
            return vmap[id(e)]
        if isinstance(e, IndexExpression):
            assert e.base.shape in ((), (1,)), "scalar variables only"
            return vmap[id(e.base)]
        if isinstance(e, BinaryOp):
            return BinaryOp(e.op, sub(e.left), sub(e.right))
        if isinstance(e, UnaryOp):
            return UnaryOp(e.op, sub(e.operand))
        if isinstance(e, FunctionCall):
            return FunctionCall(e.func_name, *[sub(a) for a in e.args])
        assert isinstance(e, Constant), type(e)
        return e

    for con in model._constraints:
        new._constraints.append(
            Constraint(body=sub(con.body), sense=con.sense, rhs=con.rhs, name=con.name)
        )
    new._objective = Objective(
        expression=sub(model._objective.expression), sense=model._objective.sense
    )
    return new


@pytest.mark.correctness
def test_st_e36_cancelling_offset_is_not_certified_infeasible():
    # On main the c = 1e6 form certified *infeasible* at the root (st_e36 is
    # feasible; minlplib optimum -246 at (5, 20)).
    base = from_nl(os.path.join(DATA, "st_e36.nl"))
    assert all(v.shape in ((), (1,)) for v in base._variables)
    model = _cancel_shift(base, [1.0e6, -1.0e6])
    assert verify_point(model, np.array([5.0, 20.0]), with_objective=True).ok
    _same_certificate(base.solve(time_limit=60), model.solve(time_limit=60))
