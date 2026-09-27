"""Regression tests for #1502: AMP on integer array variables.

A trivial integer MILP written with an array variable failed under
``solver="amp"`` in both natural spellings, while the same model with scalar
terms certified:

* ``minimize(c @ x)`` -- the canonicalizer left the scalar-valued matmul as one
  opaque atom whose aux column had no bounds and no rows, so the MILP relaxation
  was unbounded and AMP returned ``feasible`` with ``bound=None``.
* ``minimize(dm.sum(c * x))`` -- the affine linearizer walked the array-valued
  operand ``c * x`` as if it were a scalar and called ``float()`` on the vector
  ``c``; the ``TypeError`` was swallowed by a catch-all around AMP's MILP step and
  surfaced as ``status="error"``.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.canonical_expr import canonicalize
from discopt._relax.milp_relaxation import _linearize_affine_expr_sparse
from discopt.modeling.core import MatMulExpression, Model

pytestmark = pytest.mark.smoke

_C = np.array([-2.0, -1.0])


def _array_milp(form: str) -> Model:
    m = Model("amp_array_1502")
    x = m.integer("x", shape=(2,), lb=0, ub=3)
    m.subject_to(x[0] + x[1] <= 4)
    if form == "matmul":
        m.minimize(_C @ x)
    elif form == "sum":
        m.minimize(dm.sum(_C * x))
    else:
        m.minimize(-2.0 * x[0] - 1.0 * x[1])
    return m


@pytest.mark.parametrize("form", ["matmul", "sum", "scalar"])
def test_amp_certifies_integer_array_milp(form):
    """Every spelling of the same MILP certifies its optimum -7 under AMP."""
    r = _array_milp(form).solve(solver="amp", time_limit=30)
    assert r.status == "optimal", (form, r.status, r.objective, r.bound)
    assert r.gap_certified is True
    assert r.objective == pytest.approx(-7.0, abs=1e-6)
    assert r.bound is not None and r.bound == pytest.approx(-7.0, abs=1e-6)
    x = np.asarray(r.x["x"], dtype=float)
    assert x == pytest.approx([3.0, 1.0], abs=1e-6)


def test_scalar_matmul_canonicalizes_to_affine_sum():
    """``c @ x`` is the affine contraction, not an opaque atom."""
    m = _array_milp("matmul")
    dag = canonicalize(m)
    obj = dag.objective
    assert obj is not None
    assert obj.kind == "sum", obj.kind
    coeffs, const = obj.payload
    assert float(const) == 0.0
    by_var = {child.payload: float(c) for c, child in zip(coeffs, obj.children)}
    assert all(child.kind == "var" for child in obj.children)
    assert by_var == {0: -2.0, 1: -1.0}


def test_scalar_matmul_contraction_is_value_preserving():
    from discopt._relax.scalarize import scalar_matmul_contraction

    m = Model("contraction")
    x = m.continuous("x", shape=(3,), lb=-5, ub=5)
    c = np.array([1.5, -2.0, 0.25])
    expr = c @ x
    assert isinstance(expr, MatMulExpression)
    contraction = scalar_matmul_contraction(expr)
    assert contraction is not None
    coeffs, const = _linearize_affine_expr_sparse(contraction, m, 3)
    assert const == 0.0
    assert coeffs == {0: 1.5, 1: -2.0, 2: 0.25}
    # An array-valued matmul is not a scalar contraction: refused, not guessed.
    A = np.ones((2, 3))
    assert scalar_matmul_contraction(A @ x) is None


@pytest.mark.parametrize("form", ["matmul", "sum"])
def test_affine_linearizer_expands_array_objective(form):
    m = _array_milp(form)
    coeffs, const = _linearize_affine_expr_sparse(m._objective.expression, m, 2)
    assert const == 0.0
    assert coeffs == {0: -2.0, 1: -1.0}


def test_affine_linearizer_refuses_array_constant_with_value_error():
    """A vector coefficient in a scalar slot is 'not affine' (ValueError), never TypeError."""
    m = Model("vec_coeff")
    x = m.continuous("x", shape=(2,), lb=0, ub=1)
    with pytest.raises(ValueError):
        _linearize_affine_expr_sparse(_C * x, m, 2)


def test_amp_milp_build_exception_is_not_swallowed(monkeypatch):
    """An exception out of AMP's MILP build propagates instead of becoming status='error'."""
    from discopt.solvers import amp as amp_mod

    calls = []

    def broken(**kwargs):
        calls.append(1)
        raise TypeError("simulated relaxation-layer defect")

    monkeypatch.setattr(amp_mod, "_solve_milp_with_oa_recovery", broken)
    m = Model("broken_build")
    x = m.continuous("x", lb=0, ub=2)
    y = m.continuous("y", lb=0, ub=2)
    m.subject_to(x * y >= 1)
    m.minimize(x + y)
    with pytest.raises(TypeError, match="simulated relaxation-layer defect"):
        m.solve(solver="amp", time_limit=30)
    assert calls, "the patched MILP step was never reached; the test proves nothing"
