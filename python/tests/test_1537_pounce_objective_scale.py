"""#1537: POUNCE's convex lp-ipm / qp-ipm answers must not depend on objective units.

The engine's stopping test scales with the objective. A 2080-column production LP
solved with its costs in dollars, but with the same costs in cents it stopped at a
primal residual of 4.9e-6, which discopt's per-row check rejects, so the solve
returned ``error``. ``1e8*x + x**2/2`` on ``[0, 10]`` ended in
``numerical_failure``. The wrapper now solves ``sigma * objective`` with ``sigma``
a power of two and maps the answer back exactly.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sps
from discopt.solvers.convex_ipm_pounce import (
    _ENGINE_DEFAULT_TOL,
    _ENGINE_TOL_FLOOR,
    _unscale_result,
    engine_tol,
    objective_scale,
)

pytest.importorskip("pounce")


def _production_lp(T, P=3, seed=9):
    rng = np.random.default_rng(seed)
    wk = np.arange(T)
    d = rng.uniform(80, 120, (P, T)) * (1 - 0.3 * np.sin(2 * np.pi * wk / 52))
    cost = np.array([[20.0], [26.0], [31.0]])[:P] * rng.uniform(0.95, 1.05, (P, T))
    hold, a = np.array([0.8, 1.0, 1.2])[:P], np.array([1.0, 1.5, 2.0])[:P]
    H, c_ot = 400.0, 45.0
    E, S, Z = sps.eye(T, format="csr"), sps.eye(T, k=-1, format="csr"), sps.csr_matrix((T, T))
    bal = [
        [E if k == i else (S - E if k == P + i else Z) for k in range(2 * P + 2)] for i in range(P)
    ]
    cap = [a[i] * E for i in range(P)] + [Z] * P + [-E, E]
    A = sps.bmat(bal + [cap], format="csr")
    b = np.r_[d.ravel(), np.full(T, H)]
    c = np.r_[cost.ravel(), np.repeat(hold, T), np.full(T, c_ot), np.zeros(T)]
    return A, b, c


_LP = _production_lp(208)


def _solve_lp(s):
    A, b, c = _LP
    m = dm.Model("plan")
    x = m.continuous("x", shape=(A.shape[1],), lb=0)
    m.add_linear_constraints(A, x, "==", b, name="row")
    m.minimize((s * c) @ x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return m.solve(solver="pounce")


@pytest.fixture(scope="module")
def lp_reference():
    r = _solve_lp(1.0)
    assert r.status == "optimal" and r.algorithm_route == "pounce:lp-ipm"
    return r.objective


@pytest.mark.parametrize("s", [1e-2, 1.0, 100.0, 1e4, 1e6, 1e8])
def test_lp_answer_does_not_depend_on_cost_units(s, lp_reference):
    r = _solve_lp(s)
    assert r.status == "optimal", (r.status, r.error)
    assert r.algorithm_route == "pounce:lp-ipm"
    assert r.gap_certified
    assert r.objective / s == pytest.approx(lp_reference, rel=1e-9)
    assert r.bound <= r.objective


@pytest.mark.parametrize("ub", [10.0, None])
def test_qp_with_a_large_linear_coefficient(ub):
    m = dm.Model("t")
    x = m.continuous("x", lb=0, ub=ub)
    m.minimize(1e8 * x + 0.5 * x**2)
    r = m.solve(solver="pounce")
    assert r.status == "optimal", (r.status, r.error)
    assert r.algorithm_route == "pounce:qp-ipm"
    assert abs(float(r.value(x))) <= 1e-6
    assert abs(r.objective) <= 1e-6 * 1e8 * 1e-6 + 1e-9


def test_objective_scale_is_a_power_of_two_and_neutral_at_unit_scale():
    assert objective_scale(None, np.array([0.5, -1.0])) == 1.0
    assert objective_scale(None, np.zeros(3)) == 1.0
    assert objective_scale(None, np.array([], dtype=float)) == 1.0
    n_checked = 0
    for m_ in [1.5, 45.0, 4500.0, 1e8, 2.0**40]:
        sig = objective_scale(sps.csr_matrix(np.array([[m_]])), np.array([1.0]))
        mant, _ = np.frexp(sig)
        assert mant == 0.5  # exactly a power of two
        assert 0.5 < m_ * sig <= 1.0
        n_checked += 1
    assert n_checked == 5


def test_unscale_result_maps_duals_and_residuals():
    from types import SimpleNamespace

    res = SimpleNamespace(
        status="optimal",
        success=True,
        iters=3,
        x=np.array([1.0, 2.0]),
        obj=0.25,
        y=np.array([0.5]),
        z=np.array([0.125]),
        z_lb=np.array([0.0, 0.25]),
        z_ub=np.array([0.0, 0.0]),
        kkt_error=1e-9,
        residuals={
            "primal_infeasibility": 1e-9,
            "dual_infeasibility": 1e-10,
            "complementarity": 2e-10,
            "kkt_error": 1e-9,
        },
        iterates=[
            {"iter": 0, "objective": 0.5, "primal_infeasibility": 1.0, "dual_infeasibility": 0.25}
        ],
        scaling_warning=None,
    )
    assert _unscale_result(res, 1.0) is res
    u = _unscale_result(res, 0.25)
    np.testing.assert_array_equal(u.x, res.x)
    assert u.obj == 1.0
    np.testing.assert_array_equal(u.y, [2.0])
    np.testing.assert_array_equal(u.z, [0.5])
    np.testing.assert_array_equal(u.z_lb, [0.0, 1.0])
    assert u.residuals["primal_infeasibility"] == 1e-9
    assert u.residuals["dual_infeasibility"] == pytest.approx(4e-10)
    assert u.residuals["complementarity"] == pytest.approx(8e-10)
    assert u.kkt_error == pytest.approx(1e-9)  # the primal part is unscaled and still largest
    assert u.iterates[0]["objective"] == 2.0 and u.iterates[0]["primal_infeasibility"] == 1.0


# ── the engine tolerance under sigma ─────────────────────────────────────────


@pytest.mark.parametrize("k", [64 / 0.01, 1e6, 1e8])
def test_a_large_quadratic_keeps_its_caller_unit_stationarity(k):
    """``k/2*j**2 + (x-0.4)**2 - 1.6*j``, ``x - j = 0.4``: the box ``[-1e3, 1e3]``
    puts complementarity at ``z * 1e3``. Handed ``sigma*objective`` at the
    engine's absolute ``tol``, the mapped-back complementarity was ``tol/sigma``
    -- 6.6e-6 at ``k = 6400`` -- and the #1384 guard refused the point
    (``status="error"``; #1609's ``rb`` test on main after #1623)."""
    m = dm.Model("rb")
    x = m.continuous("x", lb=0, ub=1)
    j = m.continuous("j", lb=-1e3, ub=1e3)
    m.subject_to(x - j == 0.4)
    m.minimize(0.5 * k * j * j + (x - 0.4) * (x - 0.4) - 1.6 * j)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.solve(solver="pounce")
    assert r.status == "optimal", (r.status, r.error)
    assert float(r.value(j)) == pytest.approx(1.6 / (k + 2), rel=1e-5)


def test_engine_tol_is_neutral_unscaled_and_floored_scaled():
    assert engine_tol(None, 1.0) is None and engine_tol(1e-7, 1.0) == 1e-7
    assert engine_tol(None, 2.0**-4) == _ENGINE_DEFAULT_TOL * 2.0**-4
    assert engine_tol(1e-6, 2.0**-3) == 1e-6 * 2.0**-3
    assert engine_tol(None, 2.0**-40) == _ENGINE_TOL_FLOOR


def test_engine_default_tol_is_mirrored():
    """``_ENGINE_DEFAULT_TOL`` copies POUNCE's unexposed ``QpOptions`` default; if
    POUNCE changes it, ``tol=None`` and ``tol=_ENGINE_DEFAULT_TOL`` diverge here."""
    from pounce.qp import solve_qp

    P = np.array([[2.0, 0.0], [0.0, 6400.0]])
    c = np.array([-0.8, -1.6])
    kw = dict(A=np.array([[1.0, -1.0]]), b=np.array([0.4]), lb=np.array([0.0, -1e3]))
    kw["ub"] = np.array([1.0, 1e3])
    a = solve_qp(P=P, c=c, tol=None, **kw)
    b = solve_qp(P=P, c=c, tol=_ENGINE_DEFAULT_TOL, **kw)
    assert a.iters == b.iters and np.array_equal(a.x, b.x)
    # And a different tol is distinguishable, so the comparison can fail.
    d = solve_qp(P=P, c=c, tol=1e-4, **kw)
    assert d.iters != a.iters
