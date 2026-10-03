"""Issue #1582: ``dm.prod(x)`` over an array was the identity in Rust FBBT.

1. **False bound.** ``presolve/fbbt.rs`` evaluated a single-argument
   ``MathFunc::Prod`` as its argument's interval. For an array argument that
   interval is the HULL of the elements, while the node's value is the product of
   all of them. On ``p == prod(x)`` with ``x0, x1 in [1, 2]``,
   ``x2 in [-1, -0.5]`` the kernel derived ``p in [-1, 2]``; the true range is
   ``[-4, -0.5]``. The same class of defect as the ``Sum`` rule fixed in #1364.

2. **Crash.** The cut recognizer handed ``sympy.prod`` one scalar symbol for the
   whole array and raised ``TypeError: reduce() arg 2 must support iteration``
   on every solve of a model containing ``dm.prod(x)``.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._rust import model_to_repr
from discopt.tightening import fbbt_box

X_LB = np.array([1.0, 1.0, -1.0])
X_UB = np.array([2.0, 2.0, -0.5])


def _prod_model(lb, ub, *, twin: bool, sense: str = "min"):
    m = dm.Model("prod_twin" if twin else "prod")
    x = m.continuous("x", shape=(len(lb),), lb=lb, ub=ub)
    p = m.continuous("p", lb=-10, ub=10)
    if twin:
        body = x[0]
        for i in range(1, len(lb)):
            body = body * x[i]
    else:
        body = dm.prod(x)
    m.subject_to(p - body == 0)
    if sense == "min":
        m.minimize(p)
    else:
        m.maximize(p)
    return m


def test_kernel_encloses_true_range_of_prod():
    lo, hi = model_to_repr(_prod_model(X_LB, X_UB, twin=False), None).fbbt(20, 1e-9)
    # True range [-4, -0.5]; main returned [-1, 2].
    assert lo[-1] <= -4.0 + 1e-9, (lo[-1], hi[-1])
    assert hi[-1] >= -0.5 - 1e-9, (lo[-1], hi[-1])


def test_kernel_prod_feasible_point_sampling():
    """No feasible ``(x, p = prod(x))`` may be cut, across random boxes."""
    rng = np.random.default_rng(1582)
    checks = 0
    for _ in range(150):
        n = int(rng.integers(2, 6))
        a = rng.uniform(-3, 3, n)
        b = a + rng.uniform(0, 3, n)
        m = _prod_model(a, b, twin=False)
        lo, hi = model_to_repr(m, None).fbbt(20, 1e-9)
        res = fbbt_box(m)
        for _ in range(10):
            xs = rng.uniform(a, b)
            pv = float(np.prod(xs))
            if abs(pv) > 10:
                continue
            pt = np.r_[xs, pv]
            # The kernel reports one interval per variable BLOCK (x, p); the
            # tightening API reports one per scalar. Broadcast the block hull.
            assert len(lo) == 2 and len(np.asarray(res.lb)) == n + 1
            k_lo = np.r_[np.full(n, lo[0]), lo[1]]
            k_hi = np.r_[np.full(n, hi[0]), hi[1]]
            for L, U in ((k_lo, k_hi), (res.lb, res.ub)):
                assert np.all(np.asarray(L) <= pt + 1e-7), (a, b, xs, pv, L)
                assert np.all(np.asarray(U) >= pt - 1e-7), (a, b, xs, pv, U)
                checks += 1
    assert checks > 1000, f"probe did not fire: {checks} checks"


@pytest.mark.parametrize("sense", ["min", "max"])
def test_prod_solve_matches_scalar_twin(sense):
    """End-to-end: ``dm.prod(x)`` solves to the hand-written ``x0*x1*x2`` optimum."""
    r = _prod_model(X_LB, X_UB, twin=False, sense=sense).solve(time_limit=60)
    t = _prod_model(X_LB, X_UB, twin=True, sense=sense).solve(time_limit=60)
    expected = -4.0 if sense == "min" else -0.5
    assert t.status == "optimal", t.status
    assert t.objective == pytest.approx(expected, abs=1e-5)
    assert r.status == "optimal", r.status
    assert r.objective == pytest.approx(t.objective, abs=1e-5)
    # A certified bound must not cross the true optimum.
    bound = getattr(r, "bound", None)
    if bound is not None and np.isfinite(bound):
        if sense == "min":
            assert bound <= expected + 1e-5
        else:
            assert bound >= expected - 1e-5


def test_cut_recognizer_translates_prod_as_product_of_elements():
    pytest.importorskip("sympy")
    from discopt._relax.symbolic.cut_recognizer import model_to_sympy

    sm = model_to_sympy(_prod_model(X_LB, X_UB, twin=False))
    tw = model_to_sympy(_prod_model(X_LB, X_UB, twin=True))
    (_, eq), (_, eq_t) = sm.equalities[0], tw.equalities[0]
    assert (eq.lhs - eq_t.lhs).expand() == 0
    assert set(sm.symbols) == set(tw.symbols) == {"x[0]", "x[1]", "x[2]", "p"}
