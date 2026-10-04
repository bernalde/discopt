"""#1616: convexity recognition must not depend on spelling, size or round-off.

Each witness is paired with a near-miss nonconvex (or unproven) case that must
still be rejected: a convexity detector that calls a nonconvex function convex
yields a false certificate.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.convexity import Curvature, certify_convex, classify_expr, classify_model
from discopt.solvers import convex_ipm_pounce as cvx


def _laplacian(n: int) -> np.ndarray:
    """Hessian of ``sum (x[i+1] - x[i])**2``: PSD, singular (lambda_min = 0)."""
    L = np.zeros((n, n))
    for i in range(n - 1):
        L[i, i] += 2.0
        L[i + 1, i + 1] += 2.0
        L[i, i + 1] -= 2.0
        L[i + 1, i] -= 2.0
    return L


# -- A-18a / A-23: singular PSD Hessians above the exact-test cap ----------


@pytest.mark.parametrize("n", [40, 5000])
def test_singular_laplacian_hessian_is_proved_psd(n):
    # n=40 is above the dense exact cap (30); n=5000 is above the eigenvalue cap
    # (4000). No eigenvalue margin can prove lambda_min = 0.
    assert cvx.certify_psd(_laplacian(n)) is True


def test_laplacian_near_miss_is_rejected():
    # One diagonal one ulp low: lambda_min < 0 exactly (the all-ones vector).
    L = _laplacian(40)
    L[5, 5] = np.nextafter(L[5, 5], 0.0)
    assert cvx.certify_psd(L) is False
    # An off-diagonal pair one ulp too large in magnitude: also indefinite.
    L = _laplacian(40)
    L[7, 8] = L[8, 7] = np.nextafter(L[7, 8], -np.inf)
    assert cvx.certify_psd(L) is False


def test_sparse_indefinite_above_cap_is_rejected():
    L = _laplacian(60)
    L[10, 10] = -1e-12
    assert cvx.certify_psd(L) is False


def test_dense_beyond_exact_cap_still_uses_the_eigen_margin():
    rng = np.random.default_rng(0)
    n = 60
    A = rng.standard_normal((n, n))
    assert cvx.certify_psd(A.T @ A + np.eye(n)) is True
    # A dense rank-one matrix in floating point is not proved (sound).
    g = rng.standard_normal(n)
    assert cvx.certify_psd(np.outer(g, g)) is False


def test_exact_psd_budget_returns_undecided():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((40, 40))
    assert cvx._exact_psd(A.T @ A, budget=10) is None


def test_laplacian_qp_stays_on_qp_ipm():
    n = 40
    m = dm.Model("lap")
    x = m.continuous("x", shape=(n,), lb=-10, ub=10)
    f = np.sin(np.arange(n))
    f -= f.mean()
    m.minimize(
        dm.sum([(x[i + 1] - x[i]) ** 2 for i in range(n - 1)])
        - dm.sum([float(f[i]) * x[i] for i in range(n)])
    )
    m.subject_to(dm.sum([x[i] for i in range(n)]) == 0)
    r = m.solve(solver="pounce")
    assert r.algorithm_route == "pounce:qp-ipm"
    assert r.status == "optimal"


# -- B-20b: ``p**0.5`` is ``sqrt(p)`` ----------------------------------------


def _xy():
    m = dm.Model("s")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    return m, x, y


def test_pow_half_norm_is_convex_like_sqrt():
    m, x, y = _xy()
    assert classify_expr((x**2 + y**2) ** 0.5, m) == Curvature.CONVEX
    assert classify_expr(dm.sqrt(x**2 + y**2), m) == Curvature.CONVEX
    m.minimize((x**2 + y**2) ** 0.5 + 0.1 * x)
    m.subject_to(x + y >= 1)
    assert classify_model(m)[0] is True


def test_pow_half_near_misses_are_not_convex():
    m, x, y = _xy()
    assert classify_expr((x**2 - y**2) ** 0.5, m) != Curvature.CONVEX
    assert classify_expr((x**2 + y**2) ** 0.5000001, m) != Curvature.CONVEX
    # concave of convex: (x**2 + 1)**0.75 is neither convex nor concave.
    assert classify_expr((x**2 + 1) ** 0.75, m) == Curvature.UNKNOWN


def test_fractional_power_dcp_composition():
    m, x, y = _xy()
    # t**1.5 is convex nondecreasing on t >= 0; x**2 + 1 is convex and positive.
    assert classify_expr((x**2 + 1) ** 1.5, m) == Curvature.CONVEX
    p = m.continuous("p", lb=0.1, ub=4)
    # t**0.5 concave nondecreasing; sqrt(p) concave positive -> p**0.25 concave.
    assert classify_expr(dm.sqrt(p) ** 0.5, m) == Curvature.CONCAVE
    # A concave base under a convex power is not provable.
    assert classify_expr(dm.sqrt(p) ** 1.5, m) == Curvature.UNKNOWN


# -- A-01 / A-02: entropy products and the certificate ----------------------


def test_classify_model_agrees_with_solver_on_xlogx_product():
    m = dm.Model("e")
    x = m.continuous("x", lb=0.01, ub=0.99)
    m.minimize(x * dm.log(x) + 0.5 * x)
    assert classify_model(m)[0] is True


def test_entropy_product_near_misses():
    m = dm.Model("e")
    x = m.continuous("x", lb=0.01, ub=0.99)
    m.minimize(-(x * dm.log(x)))  # concave objective under min: not convex
    assert classify_model(m)[0] is False
    m2 = dm.Model("e2")
    x2 = m2.continuous("x", lb=0.01, ub=0.99)
    assert classify_expr(x2**2 * dm.log(x2), m2) != Curvature.CONVEX


def test_certify_convex_proves_xlogx_on_positive_box():
    m = dm.Model("b")
    y = m.continuous("y", lb=0.01, ub=0.99)
    assert certify_convex(dm.xlogx(y), m) == Curvature.CONVEX
    assert certify_convex(-dm.xlogx(y), m) == Curvature.CONCAVE
    # Hessian 1/y - 0.8 > 0 on the box: still convex.
    assert certify_convex(dm.xlogx(y) - 0.4 * y**2, m) == Curvature.CONVEX


def test_certify_convex_xlogx_near_misses_abstain():
    m = dm.Model("b")
    y = m.continuous("y", lb=0.01, ub=0.99)
    # Hessian 1/y - 6 changes sign on the box.
    assert certify_convex(dm.xlogx(y) - 3.0 * y**2, m) is None
    m0 = dm.Model("b0")
    y0 = m0.continuous("y", lb=0.0, ub=0.99)
    assert certify_convex(dm.xlogx(y0), m0) is None  # 1/y unbounded at 0


# -- D-23: perspective written inline ----------------------------------------


def _perspective(den_offset: float, eps: float = 1e-3):
    n = 3
    m = dm.Model("persp")
    x = m.continuous("x", shape=(n,), lb=0, ub=1)
    z = m.binary("z", shape=(n,))
    m.minimize(dm.sum(lambda i: x[i] ** 2 / ((1 - eps) * z[i] + den_offset), over=range(n)))
    m.subject_to(dm.sum(lambda i: x[i], over=range(n)) == 1)
    for i in range(n):
        m.subject_to(x[i] <= z[i])
    return m


def test_inline_perspective_is_convex():
    assert classify_model(_perspective(1e-3))[0] is True


@pytest.mark.parametrize("offset", [-1e-3, 0.0])
def test_inline_perspective_with_nonpositive_denominator_is_not_convex(offset):
    # (1-eps)*z - eps reaches -eps at z=0; (1-eps)*z reaches 0: no proof.
    assert classify_model(_perspective(offset))[0] is False
